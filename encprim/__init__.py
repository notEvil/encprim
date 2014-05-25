import struct
import cStringIO as StringIO

from pickle import encode_long, decode_long




bitarray = None

'''
syntax:
value = [count]type[data]

count (number of repetitions if not specified otherwise) is either
- 0 .. 9
- x[1 byte]
- X[4 byte]

type is one of
- N = None
-- no data
- T or F = bool
-- no data
- b, h, i or q = integer (<=8 byte)
-- 1, 2, 4 or 8 bytes
- I = big integer (<256 byte)
-- first byte in data specifies size in bytes
- J = very big integer (>=256 byte)
-- first 4 bytes in data specifies size in bytes
- d = float
- C = complex
- S = str
-- count specifies length in bytes
- : = slice
- B = bitarray
-- count specifies length in bits

tuple/list/set = sequence of values inside () / [] / {}
'''



def _encodeNone(x, visited):
    return 'N'

def _decodeNone(read, typ, n, target):
    if n == 1:
        target.append(None)
        return

    target.extend([None] * n)

def _encodeBool(x, visited):
    return 'T' if x else 'F'

def _decodeBool(read, typ, n, target):
    if n == 1:
        target.append(typ == 'T')
        return

    target.extend([typ == 'T'] * n)


def _encodeInt(x, visited, pack=struct.pack):
    pad = -1 if x < 0 else 0

    if (x >> 7) == pad:
        return 'b' + pack('b', x)

    if (x >> 15) == pad:
        return 'h' + pack('h', x)

    if (x >> 31) == pad:
        return 'i' + pack('i', x)

    if (x >> 63) == pad:
        return 'q' + pack('q', x)

    bytes = encode_long(x)
    n = len(bytes)
    if n < 256:
        return 'I' + chr(n) + bytes

    return 'J' + pack('I', n) + bytes

def _decodeInt(read, typ, n, target, size={'b': 1, 'h': 2, 'i': 4, 'q': 8}, unpack=struct.unpack):
    if n == 1:
        target.extend( unpack(typ, read(size[typ])) )
        return

    target.extend(
        unpack(str(n) + typ, read(n * size[typ]))
    )

def _decodeBigInt(read, typ, n, target):
    if n == 1:
        target.append(
            decode_long(read( ord(read(1)) ))
        )
        return

    target.extend([ decode_long(read( ord(read(1)) )) for i in xrange(n) ])

def _decodeVeryBigInt(read, typ, n, target, unpack=struct.unpack):
    if n == 1:
        target.append(
            decode_long(read( unpack('I', read(4)) ))
        )
        return

    target.extend([ decode_long(read( unpack('I', read(4))[0] )) for i in xrange(n) ])


#def _toBytes(x):
    #pad = -1 if x < 0 else 0

    #if x == pad:
        #return pad and '\xff' or '\x00'

    #r = []
    #while x != pad:
        #r.append(x & 0xff)
        #x >>= 8

    #if pad: # use last bit as sign
        #if not (r[-1] & 0x80):
            #r.append(0xff)
    #else:
        #if r[-1] & 0x80:
            #r.append(0)

    #return ''.join(map(chr, r))

#def _fromBytes(x):
    #r = -1 if ord(x[-1]) & 0x80 else 0
    #r <<= 8 * len(x)

    #i = 0
    #for c in x:
        #r |= ord(c) << i
        #i += 8

    #return r


def _encodeFloat(x, visited, pack=struct.pack):
    return 'd' + pack('d', x)

def _decodeFloat(read, typ, n, target, unpack=struct.unpack):
    if n == 1:
        target.extend( unpack('d', read(8)) )
        return

    target.extend(
        unpack(str(n) + 'd', read(n * 8))
    )

def _encodeComplex(x, visited, pack=struct.pack):
    return 'C' + pack('2d', x.real, x.imag)

def _decodeComplex(read, typ, n, target, unpack=struct.unpack):
    if n == 1:
        values = unpack('2d', read(16))
        target.append( complex(*values) )
        return

    count = 2 * n # real, imag
    values = unpack(str(count) + 'd', read(8 * count))
    target.extend( complex(real, imag) for real, imag in zip(values[::2], values[1::2]) )


def _encodeStr(x, visited):
    return _encodeCount(len(x)) + 'S' + x

def _decodeStr(read, typ, n, target):
    target.append(read(n))

def _encodeUnicode(x, visited):
    x = x.encode('utf-8')
    return _encodeCount(len(x)) + 'U' + x

def _decodeUnicode(read, typ, n, target):
    target.append( unicode(read(n), 'utf-8') )

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

def _decodeSlice(read, typ, n, target):
    decode = _decode
    if n == 1:
        r = []
        decode(read, r) # start
        decode(read, r) # stop
        decode(read, r) # step
        target.append(slice(*r))
        return

    for i in xrange(n):
        r = []
        decode(read, r) # start
        decode(read, r) # stop
        decode(read, r) # step
        target.append(slice(*r))


def _encodeCount(n, pack=struct.pack):
    if n == 1:
        return ''

    if n < 10:
        return str(n)

    if n >> 8: # doesn't fit into 1 byte
        return 'X' + pack('I', n)

    return 'x' + chr(n)


def _encodeContainer(x, visited, starts={tuple: '(', list: '[', set: '<'}, ends={tuple: ')', list: ']', set: '>'}, reducable=set(['N', 'T', 'F', 'b', 'h', 'i', 'q', 'I', 'J', 'd', 'C', ':'])):
    typ = type(x)
    r = [starts[typ]]

    if typ == set:
        x = sorted(x, key=type)

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

            r.append(_encodeCount(count) + prev)
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
            r.append(_encodeCount(count) + prev)
            if hasData:
                r.extend(data)

            break
        ## reducable

    r.append(ends[typ])

    return ''.join(r)


class EndOfContainer(Exception):
    pass

def _decodeEnd(read, typ, n, target, endEx=EndOfContainer):
    raise endEx


class Interface(object):
    def __init__(self, atts):
        self.__dict__.update(atts)

def _decodeContainer(read, typ, n, target, endEx=EndOfContainer):
    if typ == '<': # set
        r = set()
        nTarget = Interface(( ('append', lambda x, r=r: r.add(x)),
                              ('extend', lambda x, r=r: r.update(x)),)
                           )
    else: # tuple or list
        r = nTarget = []

    decode = _decode

    while True:
        try:
            decode(read, nTarget)
        except endEx:
            break

    if typ == '(': # tuple
        r = tuple(r)

    target.append(r)


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

def _decodeDict(read, typ, n, target, endEx=EndOfContainer):
    decode = _decode

    r = {}
    setitem = r.__setitem__

    while True:
        rr = []

        try:
            decode(read, rr)
        except endEx:
            break

        decode(read, rr)

        setitem(*rr)

    target.append(r)


def _encodeBitarray(x, visited):
    return _encodeCount(len(x)) + 'B' + x.tobytes()

def _decodeBitarray(read, typ, n, target):
    global bitarray
    if bitarray == None:
        import bitarray

    end = n >> 3 # l / 8
    pads = n & 0x7

    r = bitarray.bitarray()
    if pads:
        r.frombytes(read(end + 1))
        del r[-(8 - pads)::]

    else:
        r.frombytes(read(end))

    target.append(r)


encDefs = {
    type(None): _encodeNone,
    bool: _encodeBool,
    int: _encodeInt,
    long: _encodeInt,
    float: _encodeFloat,
    complex: _encodeComplex,
    str: _encodeStr,
    unicode: _encodeUnicode,
    tuple: _encodeContainer,
    list: _encodeContainer,
    set: _encodeContainer,
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
    'T': _decodeBool,
    'F': _decodeBool,
    'b': _decodeInt,
    'h': _decodeInt,
    'i': _decodeInt,
    'q': _decodeInt,
    'I': _decodeBigInt,
    'J': _decodeVeryBigInt,
    'd': _decodeFloat,
    'C': _decodeComplex,
    'S': _decodeStr,
    'U': _decodeUnicode,
    ':': _decodeSlice,
    '(': _decodeContainer,
    '[': _decodeContainer,
    '<': _decodeContainer,
    '{': _decodeDict,
    ')': _decodeEnd,
    ']': _decodeEnd,
    '>': _decodeEnd,
    '}': _decodeEnd,
    'B': _decodeBitarray,
}


def decode(file):
    r = []
    _decode(file.read, r)
    r, = r
    return r


def _decode(read, target, decDefs=decDefs):
    c = read(1)

    if '0' <= c and c <= '9':
        n = ord(c) - 48 # - ord('0')
        c = read(1)
    elif c == 'x':
        n = ord(read(1))
        c = read(1)
    elif c == 'X':
        n, = unpack('I', read(4))
        c = read(1)
    else:
        n = 1

    return decDefs[c](read, c, n, target)

def decodes(s):
    f = StringIO.StringIO(s)
    return decode(f)



## random obj generator

genFloat = lambda random: random.random() * 10 ** random.randint(0, 17)

randGens = {
    None: lambda random: None,
    bool: lambda random: random.random() < 0.5,
    1: lambda random: random.randint(-0x7f, 0x7f),
    2: lambda random: random.randint(-0x7fff, 0x7fff),
    3: lambda random: random.randint(-0x7fffff, 0x7fffff),
    4: lambda random: random.randint(-0x7fffffff, 0x7fffffff),
    5: lambda random: random.randint(-0x7fffffffff, 0x7fffffffff),
    6: lambda random: random.randint(-0x7fffffffffff, 0x7fffffffffff),
    7: lambda random: random.randint(-0x7fffffffffffff, 0x7fffffffffffff),
    8: lambda random: random.randint(-0x7fffffffffffffff, 0x7fffffffffffffff),
    16: lambda random: random.randint(-0x7fffffffffffffffffffffffffffffff, 0x7fffffffffffffffffffffffffffffff),
    float: genFloat,
    complex: lambda random: complex(genFloat(random), genFloat(random)),
}

def randPrim(minLen=0, maxLen=6, readable=True):
    import random

    n = random.randint(minLen, maxLen)
    choice = random.choice
    typs = (choice([None, bool, 1, 2, 3, 4, 5, 6, 7, 8, 16, float, complex, str, tuple, list, dict]) for i in xrange(n))

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

    import pickle as pickle

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


    import time


    def check(x, verbose=True, bitarrays=False):
        print x, '=',
        a = encode(x, bitarrays=bitarrays)

        if verbose:
            print repr(a), '(%i)' % len(a), '=',

        b = decodes(a)
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
    check( u'' )
    check( u'u' )
    check( u'uc' )
    check( slice(1, 4.5, None) )
    check( tuple() )
    x = (None, True, 1, 0.1)
    check( tuple(x) )
    check( list(x) )
    check( set(x) )
    check( (1, [2, set([3, tuple()]), set([])], []) )

    for value in True, False, 1, 2 ** 128, 2 ** (8 * 356), 3.14, complex(1.2, 3.4), slice(0, None, ()):
        check( [value] * 7 )
    check( {1: 2, 3: [None]} )

    a = []
    a.append(a)
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
    print 1, ':', compare2(1, 1000)
    print 2, ':', compare2(2, 1000)
    print 3, ':', compare2(3, 1000)
    print 4, ':', compare2(4, 1000)
    print 5, ':', compare2(5, 1000)
    print 6, ':', compare2(6, 1000)
    print 7, ':', compare2(7, 1000)
    print 8, ':', compare2(8, 1000)
    print 16, ':', compare2(16, 1000)
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
    print 1, ':', compare3(1, l, n)
    print 2, ':', compare3(2, l, n)
    print 3, ':', compare3(3, l, n)
    print 4, ':', compare3(4, l, n)
    print 5, ':', compare3(5, l, n)
    print 6, ':', compare3(6, l, n)
    print 7, ':', compare3(7, l, n)
    print 8, ':', compare3(8, l, n)
    print 16, ':', compare3(16, l, n)
    print 'float:', compare3(float, l, n)
    print 'complex:', compare3(complex, l, n)

    if bitarray != None:
        print
        print 'bitarrays'
        for i in xrange(11):
            n = 2 ** i
            print 'n =', n, ':',

            a = bitarray.bitarray(n)
            print float(len(encode(a, bitarrays=True))) / len(pickle.dumps(a, 2)) # 0.95 @ n = 8000



    n = 100
    l = 1000
    objs = [tuple(randPrim()) for i in xrange(l)]

    print
    print 'random performance'

    print 'encode:',
    begin = time.time()

    for i in xrange(n):
        for obj in objs:
            encode(obj)

    end = time.time()
    print end - begin


    print 'pickle.dumps:',
    begin = time.time()

    for i in xrange(n):
        for obj in objs:
            pickle.dumps(obj, 2)

    end = time.time()
    print end - begin


    encs = [encode(obj) for obj in objs]

    print 'decode:',
    begin = time.time()

    for i in xrange(n):
        for obj in encs:
            decodes(obj)

    end = time.time()
    print end - begin

    encs = [pickle.dumps(obj, 2) for obj in objs]

    print 'pickle.dumps:',
    begin = time.time()

    for i in xrange(n):
        for obj in encs:
            pickle.loads(obj)

    end = time.time()
    print end - begin

