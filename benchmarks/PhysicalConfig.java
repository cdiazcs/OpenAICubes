package benchmarks;

import java.io.*;
import java.lang.management.ManagementFactory;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.charset.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;


public final class PhysicalConfig {
    public final Properties evidence = new Properties();
    private final Properties config;
    private final String scenario;
    private final Path output;
    private Properties localBefore;
    private final List<String> agents = new ArrayList<>();
    private final Set<String> endpoints = new TreeSet<>();
    private String coordinator;
    private String coordinatorMachine, coordinatorMask;
    private final Map<String, String> immutable = new TreeMap<>();
    public PhysicalConfig(Properties config, Path output) { this.config = config; this.output = output; scenario = config.getProperty("scenario"); }
    public void before() throws Exception {
        if (scenario.startsWith("LOCAL_")) {
            int cores = Integer.parseInt(config.getProperty("cores"));
            localBefore = local(cores, "before");
            for (String key : localBefore.stringPropertyNames()) evidence.setProperty("local."+key, localBefore.getProperty(key));
            if (Integer.parseInt(localBefore.getProperty("selected_logical_cpus")) != cores || coreCount(localBefore.getProperty("selected_core_ids")) != cores)
                throw new IOException("physical_core_budget_not_verified");
            evidence.setProperty("verification", "local_affinity_verified");
            evidence.setProperty("compute_processes", "1");
            evidence.setProperty("observed_machines", "1");
            evidence.setProperty("compute_slots", config.getProperty("threads"));
            evidence.setProperty("experiment_processes", "1");
            evidence.setProperty("helper_processes_during_measurement", "0");
        } else {
            agents.addAll(Arrays.asList(config.getProperty("agents", "").split(",")));
            if (agents.size() != 3 || new HashSet<>(agents).size() != 3) throw new IOException("three_agents_required");
            remote("before", true);
            String endpoint = config.getProperty("host")+":"+config.getProperty("port");
            if (!endpoint.equals(coordinator)) throw new IOException("coordinator_endpoint_mismatch");
            String[] status = rpc(endpoint, "CONFIG|physical", 15000).split("\\|", -1);
            if (status.length != 8 || !status[0].equals("CONFIG") || !status[1].equals("physical")
                    || !status[2].equals(scenario) || !status[3].equals("true")
                    || !new TreeSet<>(Arrays.asList(status[5].split(","))).equals(endpoints)
                    || !status[6].equals(config.getProperty("weights.sha256"))) throw new IOException("coordinator_profile_not_verified");
            if (Integer.parseInt(status[4]) < 2 || Integer.parseInt(status[7]) != 1) throw new IOException("invalid_coordinator_chunks");
            evidence.setProperty("coordinator.config", String.join("|", status));
            Properties inventory = local(null, "client-inventory");
            if (!inventory.getProperty("machine_id").equals(coordinatorMachine)) throw new IOException("run_benchmark_on_coordinator_laptop");
            config.setProperty("client.affinity.mask", coordinatorMask);
            localBefore = local(null, "before");
            for (String key : localBefore.stringPropertyNames()) evidence.setProperty("client."+key, localBefore.getProperty(key));
            evidence.setProperty("client.role", "benchmark_control_on_coordinator_core_budget");
            evidence.setProperty("helper_processes_during_measurement", "0");
        }
        evidence.setProperty("physical_cluster_verified", "false");
    }
    public boolean after() {
        try {
            if (localBefore != null) {
                Properties after = local(null, "after");
                for (String key : after.stringPropertyNames()) evidence.setProperty("local.after."+key, after.getProperty(key));
                if (!localBefore.getProperty("affinity").equals(after.getProperty("affinity"))) throw new IOException("local_affinity_changed");
            }
            if (!scenario.startsWith("LOCAL_")) remote("after", false);
            evidence.setProperty("configuration_valid", "true"); return true;
        } catch (Exception error) { evidence.setProperty("configuration_valid", "false"); evidence.setProperty("configuration_error", error.toString()); return false; }
    }
    private Properties local(Integer cores, String suffix) throws Exception {
        String pid = ManagementFactory.getRuntimeMXBean().getName().split("@")[0];
        Path path = Paths.get(output.toString()+".physical-"+suffix+".properties").toAbsolutePath();
        if (path.getParent() != null) Files.createDirectories(path.getParent());
        List<String> args = new ArrayList<>(Arrays.asList(config.getProperty("python", "python"), "benchmarks/resource_probe.py", "--pid", pid, "--output", path.toString()));
        if (cores != null) args.addAll(Arrays.asList("--cores", cores.toString()));
        else if (suffix.equals("before") && config.containsKey("client.affinity.mask")) args.addAll(Arrays.asList("--mask", config.getProperty("client.affinity.mask")));
        Process process = new ProcessBuilder(args).redirectErrorStream(true).redirectOutput(Paths.get(output+".probe-"+suffix+".log").toFile()).start();
        if (!process.waitFor(30, TimeUnit.SECONDS)) { process.destroyForcibly(); throw new IOException("hardware_probe_timeout"); }
        if (process.exitValue() != 0) throw new IOException("hardware_probe_failed: "+output+".probe-"+suffix+".log");
        Properties result = new Properties(); try (Reader reader = Files.newBufferedReader(path, StandardCharsets.US_ASCII)) { result.load(reader); } return result;
    }
    private void remote(String phase, boolean first) throws Exception {
        Set<String> machines = new HashSet<>(); int workerCount = 0, totalSlots = 0;
        for (int at = 0; at < agents.size(); at++) {
            String response = rpc(agents.get(at), "NODEINFO|physical", 15000);
            if (!response.startsWith("NODEINFO|physical|")) throw new IOException("invalid_agent_reply");
            Properties node = new Properties(); node.load(new StringReader(response.substring("NODEINFO|physical|".length())));
            if (!node.getProperty("valid", "false").equals("true") || !node.getProperty("scenario", "").equals(scenario)
                    || !node.getProperty("weights.sha256", "").equals(config.getProperty("weights.sha256"))) throw new IOException("invalid_node_profile");
            int cores = Integer.parseInt(node.getProperty("budget.cores"));
            if ((scenario.equals("DISTRIBUTED") && cores != 1) || (scenario.equals("PARALLEL_DISTRIBUTED") && cores < 2)) throw new IOException("invalid_node_core_budget");
            if (Integer.parseInt(node.getProperty("budget.compute_threads")) != cores) throw new IOException("invalid_node_thread_budget");
            String machine = node.getProperty("machine.id"); machines.add(machine);
            int workers = Integer.parseInt(node.getProperty("workers.count")), nodeSlots = 0;
            long nodeMask = Long.parseUnsignedLong(node.getProperty("affinity.mask"), 16), union = 0;
            if (Long.bitCount(nodeMask) != cores || node.getProperty("core.ids").split(",").length != cores) throw new IOException("invalid_node_affinity");
            for (int w = 0; w < workers; w++) {
                String prefix = "worker."+w+".", endpoint = node.getProperty(prefix+"endpoint");
                long expected = Long.parseUnsignedLong(node.getProperty(prefix+"affinity.expected"), 16);
                if (!node.getProperty(prefix+"alive").equals("true") || !node.getProperty(prefix+"affinity.actual").equals(node.getProperty(prefix+"affinity.expected"))
                        || (expected & nodeMask) != expected) throw new IOException("worker_affinity_not_verified");
                String[] caps = rpc(endpoint, "CAPS|physical", 10000).split("\\|", -1);
                int slots = Integer.parseInt(node.getProperty(prefix+"slots"));
                if (caps.length != 5 || !caps[0].equals("CAPS") || !caps[1].equals("physical") || !caps[2].equals(node.getProperty(prefix+"language"))
                        || Integer.parseInt(caps[4]) != slots || !Arrays.asList(caps[3].split(",")).contains("POST")) throw new IOException("worker_capability_mismatch");
                if (slots < 1 || slots > Long.bitCount(expected)) throw new IOException("too_many_compute_slots");
                if (caps[2].equals("Python") && slots != 1) throw new IOException("gil_threads_not_cpu_parallel");
                nodeSlots += slots; union |= expected;
                if (first && !endpoints.add(endpoint)) throw new IOException("duplicate_worker_endpoint");
            }
            if (nodeSlots != cores || union != nodeMask || (scenario.equals("DISTRIBUTED") && workers != 1)) throw new IOException("invalid_node_worker_budget");
            String coord = node.getProperty("coordinator.endpoint");
            if (coord != null) {
                if (coordinator != null && !coordinator.equals(coord)) throw new IOException("multiple_coordinators");
                coordinator = coord;
                if (first) { coordinatorMachine = machine; coordinatorMask = node.getProperty("affinity.mask"); }
                if (!node.getProperty("coordinator.affinity").equals(node.getProperty("affinity.mask"))) throw new IOException("coordinator_affinity_not_verified");
                if (first) immutable.put("registered", node.getProperty("coordinator.registered"));
            }
            workerCount += workers; totalSlots += nodeSlots;
            for (String key : node.stringPropertyNames()) {
                String full = "node."+at+"."+key;
                evidence.setProperty(phase+"."+full, node.getProperty(key));
                if (key.equals("machine.id") || key.endsWith(".pid") || key.endsWith("endpoint") || key.equals("affinity.mask")) {
                    if (first) immutable.put(full, node.getProperty(key));
                    else if (!Objects.equals(immutable.get(full), node.getProperty(key))) throw new IOException("node_changed_during_run");
                }
            }
        }
        if (first && !new TreeSet<>(Arrays.asList(immutable.getOrDefault("registered", "").split(","))).equals(endpoints)) throw new IOException("registered_workers_mismatch");
        boolean local = machines.size() != 3;
        if (local && !Boolean.parseBoolean(config.getProperty("allow.localhost", "false"))) throw new IOException("three_distinct_machines_required");
        evidence.setProperty("verification", local ? "localhost_emulation" : "remote_machine_ids_observed_physical_pending");
        evidence.setProperty("observed_machines", Integer.toString(machines.size()));
        evidence.setProperty("compute_processes", Integer.toString(workerCount)); evidence.setProperty("compute_slots", Integer.toString(totalSlots));
        evidence.setProperty("experiment_processes", Integer.toString(workerCount+5));
    }
    private static int coreCount(String list) { String value = list.replace("[", "").replace("]", "").trim(); return value.isEmpty() ? 0 : value.split(",").length; }
    public static String rpc(String endpoint, String message, int timeout) throws IOException {
        int split = endpoint.lastIndexOf(':'); if (split < 1) throw new IOException("invalid_endpoint");
        try (Socket socket = new Socket()) {
            AtomicBoolean expired = new AtomicBoolean();
            Timer guard = new Timer("benchmark-deadline", true);
            guard.schedule(new TimerTask() { public void run() { expired.set(true); try { socket.close(); } catch (IOException ignored) { } } }, timeout);
            try {
                socket.connect(new InetSocketAddress(endpoint.substring(0, split), Integer.parseInt(endpoint.substring(split+1))), Math.min(5000, timeout));
                socket.setSoTimeout(timeout); socket.setTcpNoDelay(true);
                byte[] body = message.getBytes(StandardCharsets.UTF_8);
                DataOutputStream out = new DataOutputStream(socket.getOutputStream()); out.writeInt(body.length); out.write(body); out.flush();
                DataInputStream in = new DataInputStream(socket.getInputStream()); int length = in.readInt();
                if (length <= 0 || length > 2_000_000) throw new IOException("invalid_response_length");
                byte[] data = new byte[length]; in.readFully(data);
                return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(data)).toString();
            } catch (IOException error) {
                if (expired.get()) throw new SocketTimeoutException("absolute_deadline_exceeded");
                throw error;
            } finally { guard.cancel(); }
        }
    }
}
