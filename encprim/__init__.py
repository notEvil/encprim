import struct
import re





bitarray = None

'''
syntax:
[n] = repitition (int)
[l] = len (in bytes if not specified otherwise)
[d] = data
- [n]N = None
- [n]T, [n]F = bool
- [n]b[d], [n]h[d], [n]i[d], [n]q[d] = integer (var size)
- [l]Q[d] = large integer (var size)
- [n]d[d] = float
- [n]C[d] = complex
- [l]s[d] = str
- [n]:[d] = slice
- () = tuple
- [] = list
- <> = set
- {} = dict
- [l]B[d] = bitarray
  l in bits
'''



def _encodeNone(x, visited):
    return 'N'

def _decodeNone(x, n, target):
    target.extend([None] * n)
    return x

def _encodeBool(x, visited):
    return 'T' if x else 'F'

def _decodeBool(x, n, target, value):
    target.extend([value] * n)
    return x


def _encodeNum(x, visited):
    encodeAs = _encodeNumAs

    r = encodeAs(x, 'b')
    if r != None:
        return r

    r = encodeAs(x, 'h')
    if r != None:
        return r

    r = encodeAs(x, 'i')
    if r != None:
        return r

    r = encodeAs(x, 'q')
    if r != None:
        return r

    r = _toBytes(x)
    return str(len(r)) + 'Q' + r

def _encodeNumAs(x, format):
    try:
        return format + struct.pack(format, x)
    except struct.error:
        return None


def _decodeNum(x, n, target, typ, size={'b': 1, 'h': 2, 'i': 4, 'q': 8}):
    if typ == 'Q':
        target.append(
            _fromBytes(buffer(x, 0, n))
        )
        return buffer(x, n)

    l = n * size[typ]
    target.extend(
        struct.unpack(str(n) + typ, buffer(x, 0, l))
    )
    return buffer(x, l)


def _toBytes(x, lastBit=0x80):
    if x == 0:
        return ''

    if x < 0:
        neg = True
        x = abs(x)
    else:
        neg = False

    r = []

    while x:
        r.append(x & 0xff)
        x >>= 8

    if neg:
        last = r[-1]
        if last & lastBit:
            r.append(lastBit)
        else:
            r[-1] = last | lastBit

    else:
        if r[-1] & lastBit:
            r.append(0)

    return ''.join(chr(x) for x in r)

def _fromBytes(x, lastBit=0x80):
    if len(x) == 0:
        return 0

    x = map(ord, x)

    last = x[-1]
    if last & lastBit:
        neg = True

        x[-1] = last ^ lastBit
    else:
        neg = False

    r = 0
    i = 0
    for item in x:
        r |= (item << i)
        i += 8

    return -r if neg else r


def _encodeFloat(x, visited):
    return 'd' + struct.pack('d', x)

def _decodeFloat(x, n, target):
    l = n * 8
    target.extend(
        struct.unpack(str(n) + 'd', buffer(x, 0, l))
    )
    return buffer(x, l)

def _encodeComplex(x, visited):
    return 'C' + struct.pack('2d', x.real, x.imag)

def _decodeComplex(x, n, target):
    count = 2 * n # real, imag
    l = 8 * count # 8 byte per float
    values = struct.unpack(str(count) + 'd', buffer(x, 0, l))
    target.extend( complex(real, imag) for real, imag in zip(values[::2], values[1::2]) )
    return buffer(x, l)


def _encodeStr(x, visited):
    l = len(x)
    return ('' if l == 1 else str(l)) + 'S' + x

def _decodeStr(x, l, target):
    target.append(x[:l])
    return buffer(x, l)

def _encodeSlice(x, visited):
    encode = _encode

    start = encode(x.start, visited)
    if start == None:
        return None

    stop = encode(x.stop, visited)
    if stop == None:
        return None

    step = encode(x.step, visited)
    if step == None:
        return None

    return ':' + start + stop + step

def _decodeSlice(x, n, target):
    decode = _decode
    for i in xrange(n):
        r = []
        x = decode(x, r) # start
        x = decode(x, r) # stop
        x = decode(x, r) # step
        target.append(slice(*r))

    return x



def _encodeContainer(x, visited, start, end, reducable=set(['N', 'T', 'F', 'b', 'h', 'i', 'q', 'd', 'C', ':'])):
    r = [start]

    iX = iter(x)

    encode = _encode

    prev = ''

    for item in iX:
        ## not reducable
        n = encode(item, visited)
        if n == None:
            return None

        nPrev = n[0]
        if nPrev == prev:
            r.append(n)
            continue

        prev = nPrev
        if prev not in reducable:
            r.append(n)
            continue
        ## not reducable

        ## reducable
        count = 1

        if len(n) != 1:
            hasData = True
            data = [n[1::]]
        else:
            hasData = False

        for item in iX:
            n = encode(item, visited)
            if n == None:
                return None

            nPrev = n[0]
            if nPrev == prev:
                count += 1

                if hasData:
                    data.append(n[1::])
                continue

            r.append(prev if count == 1 else str(count) + prev)
            if hasData:
                r.extend(data)

            prev = nPrev
            if prev in reducable:
                count = 1

                if len(n) != 1:
                    hasData = True
                    data = [n[1::]]
                else:
                    hasData = False

                continue

            r.append(n) # first not reducable
            break

        else: # out of items
            r.append(prev if count == 1 else str(count) + prev)
            if hasData:
                r.extend(data)

            break
        ## reducable

    r.append(end)

    return ''.join(r)


class EndOfContainer(Exception):
    pass

def _decodeEnd(x, n, target, endEx=EndOfContainer):
    raise endEx

def _decodeContainer(x, target, typ, endEx=EndOfContainer):
    r = []

    decode = _decode

    while True:
        try:
            x = decode(x, r)
        except endEx:
            break

    if typ != list:
        r = typ(r)

    target.append(r)
    return buffer(x, 1)


def _encodeDict(x, visited):
    encode = _encode

    r = ['{']

    for key, value in x.iteritems():
        key = encode(key, visited)
        if key == None:
            return None

        r.append(key)

        value = encode(value, visited)
        if value == None:
            return None

        r.append(value)

    r.append('}')

    return ''.join(r)

def _decodeDict(x, n, target, endEx=EndOfContainer):
    decode = _decode

    r = []

    while True:
        rr = []

        try:
            x = decode(x, rr)
        except endEx:
            break

        x = decode(x, rr)

        r.append(rr)

    target.append(dict(r))
    return buffer(x, 1)


def _encodeBitarray(x, visited):
    return str(len(x)) + 'B' + x.tobytes()

def _decodeBitarray(x, l, target):
    global bitarray
    if bitarray == None:
        import bitarray

    end = l >> 3 # l / 8
    pads = l & 0x7

    r = bitarray.bitarray()
    if pads:
        end += 1
        r.frombytes(x[:end]) # frombytes needs string instead of buffer
        del r[-(8 - pads)::]

    else:
        r.frombytes(x[:end]) # see above

    target.append(r)

    return buffer(x, end)


encDefs = {
    type(None): _encodeNone,
    bool: _encodeBool,
    int: _encodeNum,
    long: _encodeNum,
    float: _encodeFloat,
    complex: _encodeComplex,
    str: _encodeStr,
    tuple: lambda x, visited, enc=_encodeContainer: enc(x, visited, '(', ')'),
    list: lambda x, visited, enc=_encodeContainer: enc(x, visited, '[', ']'),
    set: lambda x, visited, enc=_encodeContainer: enc(sorted(x, key=type), visited, '<', '>'),
    dict: _encodeDict,
    slice: _encodeSlice,
}


def encode(x, bitarrays=False, encDefs=encDefs):
    global bitarray
    if bitarrays:
        if bitarray == None:
            import bitarray

        encDefs[bitarray.bitarray] = _encodeBitarray

    elif bitarray != None:
        encDefs.pop(bitarray.bitarray, None)

    return _encode(x, set())


def _encode(x, visited, encDefs=encDefs):
    xId = id(x)
    if xId in visited:
        return None
    visited.add(xId)

    f = encDefs.get(type(x), None)
    if f == None:
        return None

    r = f(x, visited)

    visited.remove(xId)
    return r


decDefs = {
    'N': _decodeNone,
    'T': lambda x, n, target, dec=_decodeBool: dec(x, n, target, True),
    'F': lambda x, n, target, dec=_decodeBool: dec(x, n, target, False),
    'b': lambda x, n, target, dec=_decodeNum: dec(x, n, target, 'b'),
    'h': lambda x, n, target, dec=_decodeNum: dec(x, n, target, 'h'),
    'i': lambda x, n, target, dec=_decodeNum: dec(x, n, target, 'i'),
    'q': lambda x, n, target, dec=_decodeNum: dec(x, n, target, 'q'),
    'Q': lambda x, n, target, dec=_decodeNum: dec(x, n, target, 'Q'),
    'd': _decodeFloat,
    'C': _decodeComplex,
    'S': _decodeStr,
    ':': _decodeSlice,
    '(': lambda x, n, target, dec=_decodeContainer: dec(x, target, tuple),
    '[': lambda x, n, target, dec=_decodeContainer: dec(x, target, list),
    '<': lambda x, n, target, dec=_decodeContainer: dec(x, target, set),
    '{': _decodeDict,
    ')': _decodeEnd,
    ']': _decodeEnd,
    '>': _decodeEnd,
    '}': _decodeEnd,
    'B': _decodeBitarray,
}


def decode(x):
    r = []
    _decode(x, r)
    r, = r
    return r

def _decode(x, target, pattern=re.compile('(\d*)(.)')):
    match = pattern.match(x)

    n, c = match.groups()
    n = 1 if len(n) == 0 else int(n)

    x = buffer(x, match.end())

    f = decDefs.get(c, None)
    if f == None:
        raise TypeError('unknown type "%s"' % c)

    return f(x, n, target)



## random obj generator

genFloat = lambda random: random.random() * 10 ** random.randint(0, 17)

randGens = {
    None: lambda random: None,
    bool: lambda random: random.random() < 0.5,
    'i': lambda random: random.randint(-0x80000000, 0x7fffffff),
    'I': lambda random: random.randint(-0x8000000000000000, 0x7fffffffffffffff),
    'Q': lambda random: random.getrandbits(random.randint(64, 128)),
    float: genFloat,
    complex: lambda random: complex(genFloat(random), genFloat(random)),
}

def randPrim(minLen=0, maxLen=6, readable=True):
    import random

    n = random.randint(minLen, maxLen)
    choice = random.choice
    typs = (choice([None, bool, 'i', 'I', 'Q', float, complex, str, tuple, list, dict]) for i in xrange(n))

    randPrim = _randPrim
    nMaxLen = max(minLen, maxLen - 1)
    return (randPrim(typ, minLen=minLen, maxLen=nMaxLen, readable=readable) for typ in typs)




def _randPrim(typ, minLen=0, maxLen=6, readable=True):
    import random

    f = randGens.get(typ, None)
    if f != None:
        return f(random)

    if typ == str:
        n = random.randint(minLen, maxLen)
        if n == 0:
            return ''

        r = []
        while True:
            c = chr(random.randint(0, 255))
            if not readable or len(repr(c)) == 3:
                r.append(c)

                if len(r) == n:
                    break

        return ''.join(r)

    _randPrim = randPrim

    if typ in [tuple, list]:
        return typ(_randPrim(minLen=minLen, maxLen=maxLen, readable=readable))

    n = random.randint(minLen, maxLen)
    nMinLen = max(minLen, 1)
    nMaxLen = max(nMinLen, maxLen - 1)

    if typ == set:
        r = set()
        if n == 0:
            return r

        while True:
            for x in _randPrim(minLen=nMinLen, maxLen=nMaxLen, readable=readable):
                try:
                    r.add(x)
                except TypeError:
                    continue

                if len(r) == n:
                    break
            else: # n not yet reached
                continue

            break

        return r

    if typ == dict:
        r = {}
        if n == 0:
            return r

        while True:
            # get key
            while True:
                for x in _randPrim(minLen=nMinLen, maxLen=nMaxLen, readable=readable):
                    try:
                        hash(x)
                    except TypeError:
                        continue

                    break
                else: # no key found
                    continue

                key = x
                break

            # get value
            while True:
                try:
                    value = _randPrim(minLen=nMinLen, maxLen=nMaxLen, readable=readable).next()
                except StopIteration:
                    continue

                break

            r[key] = value

            if len(r) == n:
                break

        return r

    return None

## random obj generator





if __name__ == '__main__':
    try:
        import bitarray
    except:
        pass

    import pickle


    import sys
    if '-i' in sys.argv:
        if bitarray != None:
            print 'bitarray available as ba'
            ba = bitarray.bitarray

        print 'Press q to exit'

        while True:
            n = raw_input('= ')

            if n == 'q':
                sys.exit()
            elif n == '':
                continue
            elif n == 'h' or n == 'help' or n == '?':
                print 'Press q to exit'

            try:
                n = eval(n)
            except Exception, e:
                print 'Error:'
                print e
                continue

            a = encode(n, bitarrays=bitarray != None)
            if a == None:
                print 'Warning: is not primitive'
                continue
            b = pickle.dumps(n, 2)

            print repr(a), '(%s)' % len(a)
            print float(len(a)) / len(b)



    def check(x, verbose=True, bitarrays=False):
        print x, '=',
        a = encode(x, bitarrays=bitarrays)

        if verbose:
            print repr(a), '(%i)' % len(a), '=',

        b = decode(a)
        print b

        assert x == b

    check( None )
    check( True )
    check( False )
    check( 0 )
    check( 0x7f )
    check( 0x7f + 1 )
    check( 0x7fff )
    check( 0x7fff + 1 )
    check( 0x7fffffff )
    check( 0x7fffffff + 1 )
    check( 0x7fffffffffffffff )
    check( 0x7fffffffffffffff + 1 )
    check( 0x7fffffffffffffffffffffffffffffff )
    check( -1 )
    check( -0x80 )
    check( -0x80 - 1 )
    check( -0x8000 )
    check( -0x8000 - 1 )
    check( -0x80000000 )
    check( -0x80000000 - 1 )
    check( -0x8000000000000000 )
    check( -0x8000000000000000 - 1 )
    check( -0x80000000000000000000000000000000 )
    check(3.141521)
    check( complex(1.2, 3.4) )
    check( '' )
    check( 'a' )
    check( 'ab' )
    check( slice(1, 4.5, None) )
    check( tuple() )
    x = (None, True, 1, 0.1)
    check( tuple(x) )
    check( list(x) )
    check( set(x) )
    check( (1, [2, set([3, tuple()]), set([])], []) )

    for value in True, False, 1, 3.14, complex(1.2, 3.4), slice(0, None, ()):
        check( [value] * 7 )
    check( {1: 2, 3: [None]} )

    a = []
    a.append(a)
    assert encode( u'' ) == None
    assert encode( a ) == None

    if bitarray != None:
        check( bitarray.bitarray([1, 0, 0, 1]), bitarrays = True)
        check( bitarray.bitarray([1, 0, 0, 1, 0, 0, 1, 0]), bitarrays = True)
        check( bitarray.bitarray([1, 0, 0, 1, 0, 0, 1, 0, 1, 1]), bitarrays = True)



    import random

    def compare(n, l):
        pSum = 0.

        for i in xrange(n):
            obj = tuple(randPrim(0, l))

            a = len(encode(obj))
            b = len(pickle.dumps(obj, 2))

            p = float(a) / b

            pSum += p

        return pSum / n

    n = 1000
    print
    print 'random prims'
    print 's = 2:', compare(n, 2)
    print 's = 4:', compare(n, 4)
    print 's = 6:', compare(n, 6)
    print 's = 8:', compare(n, 8)
    print 's = 10:', compare(n, 10)
    print 's = 12:', compare(n, 12)


    def compare2(typ, n):
        pSum = 0.
        f = randGens[typ]

        for i in xrange(n):
            obj = f(random)

            a = len(encode(obj))
            b = len(pickle.dumps(obj, 2))

            pSum += float(a) / b

        return pSum / n


    print
    print 'random very prims'
    print 'None:', compare2(None, 1000)
    print 'bool:', compare2(bool, 1000)
    print 'i:', compare2('i', 1000)
    print 'I:', compare2('I', 1000)
    print 'Q:', compare2('Q', 1000)
    print 'float:', compare2(float, 1000)
    print 'complex:', compare2(complex, 1000)

    def compare3(typ, l, n):
        f = randGens[typ]

        pSum = 0.
        for i in xrange(n):
            obj = tuple(f(random) for i in xrange(l))

            a = len(encode(obj))
            b = len(pickle.dumps(obj, 2))

            pSum += float(a) / b

        return pSum / n

    n = 1000
    l = 32

    print
    print 'random flat, typed prims'
    print 'None:', compare3(None, l, n)
    print 'bool:', compare3(bool, l, n)
    print 'i:', compare3('i', l, n)
    print 'I:', compare3('I', l, n)
    print 'Q:', compare3('Q', l, n)
    print 'float:', compare3(float, l, n)
    print 'complex:', compare3(complex, l, n)

