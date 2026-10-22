import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys


def topology():
    if sys.platform == 'win32':
        import winreg
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError('Python de 64 bits requerido')
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        size = ctypes.c_uint32()
        kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not kernel.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        cores, offset = [], 0
        while offset < size.value:
            relation, length = struct.unpack_from('<II', buffer.raw, offset)
            count = struct.unpack_from('<H', buffer.raw, offset+30)[0]
            mask, group = struct.unpack_from('<QH', buffer.raw, offset+32)
            if relation != 0 or length < 48 or count != 1 or group != 0:
                raise RuntimeError('processor_groups_not_supported')
            cores.append({'id': len(cores), 'mask': mask, 'efficiency': buffer.raw[offset+9]})
            offset += length
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Cryptography') as key:
            identity = winreg.QueryValueEx(key, 'MachineGuid')[0]
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    elif sys.platform.startswith('linux'):
        groups = {}
        for path in sorted(Path('/sys/devices/system/cpu').glob('cpu[0-9]*')):
            index = int(path.name[3:])
            key = tuple((path/'topology'/name).read_text().strip() for name in ('physical_package_id', 'core_id'))
            groups[key] = groups.get(key, 0) | (1 << index)
        cores = [{'id': i, 'mask': mask} for i, mask in enumerate(groups.values())]
        identity = Path('/etc/machine-id').read_text().strip()
        cpu = platform.processor() or platform.machine()
    else:
        raise RuntimeError('unsupported_affinity_platform')
    if not cores:
        raise RuntimeError('cpu_topology_unavailable')
    return {'machine_id': hashlib.sha256(identity.encode()).hexdigest(), 'hostname': platform.node(),
            'os': platform.platform(), 'cpu': cpu, 'physical_cores': len(cores),
            'logical_cpus': sum(bin(c['mask']).count('1') for c in cores), 'cores': cores}


def handle(pid, write=False):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    value = kernel.OpenProcess(0x0400 | (0x0200 if write else 0), False, pid)
    if not value:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel, value


def affinity(pid):
    if sys.platform == 'win32':
        kernel, process = handle(pid)
        result, system = ctypes.c_size_t(), ctypes.c_size_t()
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        try:
            if not kernel.GetProcessAffinityMask(process, ctypes.byref(result), ctypes.byref(system)):
                raise ctypes.WinError(ctypes.get_last_error())
            return result.value
        finally:
            kernel.CloseHandle(process)
    return sum(1 << cpu for cpu in os.sched_getaffinity(pid))


def set_affinity(pid, mask):
    if not mask:
        raise ValueError('empty_affinity')
    if sys.platform == 'win32':
        kernel, process = handle(pid, True)
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        try:
            if not kernel.SetProcessAffinityMask(process, mask):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(process)
    else:
        os.sched_setaffinity(pid, {i for i in range(mask.bit_length()) if mask & (1 << i)})
    if affinity(pid) != mask:
        raise RuntimeError('affinity_not_applied')


def select_cores(info, count, start=0, allowed=None):
    if count < 1 or start < 0:
        raise ValueError('invalid_core_count')
    available = []
    for core in info['cores']:
        mask = core['mask'] if allowed is None else core['mask'] & allowed
        if mask:
            available.append((core['id'], mask & -mask))
    selected = available[start:start+count]
    if len(selected) != count:
        raise RuntimeError('not_enough_physical_cores')
    return selected


def properties(data):
    def escape(raw):
        text = str(raw).replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r')
        return ''.join(c if 32 <= ord(c) < 127 else '\\u%04x' % ord(c) for c in text)
    return ''.join(str(key)+'='+escape(value)+'\n' for key, value in data.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, default=os.getpid())
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--cores', type=int)
    selection.add_argument('--mask', type=lambda value: int(value, 16))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = topology()
    if args.mask is not None:
        set_affinity(args.pid, args.mask)
    if args.cores is not None:
        selected = select_cores(data, args.cores, allowed=affinity(args.pid))
        set_affinity(args.pid, sum(mask for _, mask in selected))
    mask = affinity(args.pid)
    data.update(pid=args.pid, affinity=format(mask, 'x'), selected_logical_cpus=bin(mask).count('1'),
                selected_core_ids=[c['id'] for c in data['cores'] if c['mask'] & mask])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(properties({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in data.items()}), encoding='ascii')


if __name__ == '__main__':
    main()
