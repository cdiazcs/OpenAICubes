
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import zlib

PNG = b'\x89PNG\r\n\x1a\n'
MAX_PIXELS = 4_194_304
MAX_FILE = 32_000_000


@dataclass(frozen=True)
class Raster:
    width: int
    height: int
    pixels: tuple


def dimensions(width, height):
    if (type(width) is not int or type(height) is not int
            or width < 1 or height < 1 or width*height > MAX_PIXELS):
        raise ValueError('image_dimensions')


def _chunk(kind, body):
    return struct.pack('!I', len(body)) + kind + body + struct.pack('!I', zlib.crc32(kind+body))


def encode_png(width, height, pixels):
    dimensions(width, height)
    if len(pixels) != width*height or any(type(x) is not int or not 0 <= x <= 255 for x in pixels):
        raise ValueError('image_samples')
    data = b''.join(b'\0' + bytes(pixels[r*width:(r+1)*width]) for r in range(height))
    header = struct.pack('!IIBBBBB', width, height, 8, 0, 0, 0, 0)
    return PNG + _chunk(b'IHDR', header) + _chunk(b'IDAT', zlib.compress(data, 9)) + _chunk(b'IEND', b'')


def _paeth(a, b, c):
    p = a+b-c
    distances = (abs(p-a), abs(p-b), abs(p-c))
    return (a, b, c)[distances.index(min(distances))]


def _png(data):
    offset, header, compressed, palette, transparency, ended = 8, None, bytearray(), None, b'', False
    while offset < len(data):
        if offset+12 > len(data):
            raise ValueError('truncated_png_chunk')
        size = struct.unpack_from('!I', data, offset)[0]
        kind = data[offset+4:offset+8]
        end = offset+8+size
        if end+4 > len(data) or zlib.crc32(data[offset+4:end]) != struct.unpack_from('!I', data, end)[0]:
            raise ValueError('png_chunk_or_crc')
        body = data[offset+8:end]
        offset = end+4
        if header is None and kind != b'IHDR':
            raise ValueError('png_header_first')
        if kind == b'IHDR':
            if header is not None or len(body) != 13:
                raise ValueError('png_header')
            header = struct.unpack('!IIBBBBB', body)
        elif kind == b'IDAT':
            compressed.extend(body)
        elif kind == b'PLTE':
            palette = body
        elif kind == b'tRNS':
            transparency = body
        elif kind == b'IEND':
            if body or offset != len(data):
                raise ValueError('png_end')
            ended = True
            break
        elif not kind[0] & 32:
            raise ValueError('unsupported_png_chunk')
    if header is None or not ended:
        raise ValueError('incomplete_png')
    width, height, depth, color, compression, filtering, interlace = header
    dimensions(width, height)
    if depth != 8 or color not in (0, 2, 3, 4, 6) or (compression, filtering, interlace) != (0, 0, 0):
        raise ValueError('supported_png_8bit_noninterlaced_only')
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    stride = width*channels
    expected = height*(stride+1)
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(bytes(compressed), expected+1)
    except zlib.error as error:
        raise ValueError('png_compression') from error
    if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('png_decoded_size')
    previous = bytearray(stride)
    samples = bytearray()
    for r in range(height):
        mode = raw[r*(stride+1)]
        row = bytearray(raw[r*(stride+1)+1:(r+1)*(stride+1)])
        if mode > 4:
            raise ValueError('png_filter')
        for i in range(stride):
            left = row[i-channels] if i >= channels else 0
            above = previous[i]
            corner = previous[i-channels] if i >= channels else 0
            predictor = (0, left, above, (left+above)//2, _paeth(left, above, corner))[mode]
            row[i] = (row[i]+predictor) & 255
        samples.extend(row)
        previous = row
    if color == 3 and (not palette or len(palette) % 3 or len(palette) > 768):
        raise ValueError('png_palette')
    pixels = []
    for i in range(0, len(samples), channels):
        parts = samples[i:i+channels]
        alpha = 255
        if color in (0, 4):
            red = green = blue = parts[0]
            if color == 4:
                alpha = parts[1]
            elif transparency == struct.pack('!H', parts[0]):
                alpha = 0
        elif color == 3:
            index = parts[0]
            if index*3+3 > len(palette):
                raise ValueError('png_palette_index')
            red, green, blue = palette[index*3:index*3+3]
            alpha = transparency[index] if index < len(transparency) else 255
        else:
            red, green, blue = parts[:3]
            if color == 6:
                alpha = parts[3]
            elif transparency == struct.pack('!HHH', red, green, blue):
                alpha = 0
        pixels.append((299*red+587*green+114*blue)/1000 * alpha/255)
    return Raster(width, height, tuple(pixels))


def _pgm(data):
    position = 0

    def token():
        nonlocal position
        while position < len(data):
            if data[position] in b' \r\n\t\v\f':
                position += 1
            elif data[position] == 35:
                while position < len(data) and data[position] not in b'\r\n':
                    position += 1
            else:
                break
        start = position
        while position < len(data) and data[position] not in b' \r\n\t\v\f#':
            position += 1
        if start == position:
            raise ValueError('pgm_token')
        return data[start:position]

    magic = token()
    width, height, maximum = int(token()), int(token()), int(token())
    dimensions(width, height)
    if not 0 < maximum <= 255:
        raise ValueError('supported_pgm_8bit_only')
    if magic == b'P5':
        if position >= len(data) or data[position] not in b' \r\n\t\v\f':
            raise ValueError('pgm_separator')
        position += 2 if data[position:position+2] == b'\r\n' else 1
        values = data[position:]
        if len(values) != width*height:
            raise ValueError('pgm_sample_count')
    elif magic == b'P2':
        values = [int(token()) for _ in range(width*height)]
        # Solo se permiten comentarios/espacios despues del ultimo pixel.
        if data[position:].strip():
            try:
                token()
            except ValueError:
                pass
            else:
                raise ValueError('pgm_extra_samples')
    else:
        raise ValueError('pgm_magic')
    if any(value < 0 or value > maximum for value in values):
        raise ValueError('pgm_sample_range')
    return Raster(width, height, tuple(value*255.0/maximum for value in values))


def decode_image(data):
    if not isinstance(data, bytes) or len(data) > MAX_FILE:
        raise ValueError('image_file_size')
    if data.startswith(PNG):
        return _png(data)
    if data.startswith((b'P2', b'P5')):
        return _pgm(data)
    raise ValueError('supported_formats_png_pgm')


def normalize(raster, size=16):
    dimensions(raster.width, raster.height)
    dimensions(size, size)
    if len(raster.pixels) != raster.width*raster.height or not all(
            math.isfinite(x) and 0 <= x <= 255 for x in raster.pixels):
        raise ValueError('image_samples')
    return tuple(raster.pixels[min(raster.height-1, int((r+.5)*raster.height/size))*raster.width
                               + min(raster.width-1, int((c+.5)*raster.width/size))]/255
                 for r in range(size) for c in range(size))


def read_image(path):
    path = Path(path)
    if path.stat().st_size > MAX_FILE:
        raise ValueError('image_file_size')
    return normalize(decode_image(path.read_bytes()))
