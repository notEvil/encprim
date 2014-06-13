import struct
import cStringIO as StringIO

from pickle import encode_long, decode_long
import itertools as it


try:
    import bitarray
except:
    bitarray = None



'''
syntax:
value = [count]type[data]

count (number of repetitions if not specified otherwise) is either
- 0 .. 9
- p[1 byte]
- P[4 byte]

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


encDefs = {}
decDefs = {}


def _encodeNone(x):
    return 'N'
encDefs[type(None)] = _encodeNone

def _decodeNone(read, n, typ):
    if n == 1:
        return None

    return [None] * n
decDefs['N'] = _decodeNone

def _encodeBool(x):
    return 'T' if x else 'F'
encDefs[bool] = _encodeBool

def _decodeBool(read, n, typ):
    if n == 1:
        return typ == 'T'

    return [typ == 'T'] * n
decDefs['T'] = _decodeBool
decDefs['F'] = _decodeBool


def _encodeInt(x, pack=struct.pack):
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
encDefs[int] = _encodeInt
encDefs[long] = _encodeInt

def _decodeInt(read, n, typ, size={'b': 1, 'h': 2, 'i': 4, 'q': 8}, unpack=struct.unpack):
    if n == 1:
        return unpack(typ, read(size[typ]))[0]

    return unpack(str(n) + typ, read(n * size[typ]))
decDefs['b'] = _decodeInt
decDefs['h'] = _decodeInt
decDefs['i'] = _decodeInt
decDefs['q'] = _decodeInt

def _decodeBigInt(read, n, typ):
    if n == 1:
        return decode_long(read( ord(read(1)) ))

    return [ decode_long(read( ord(read(1)) )) for i in xrange(n) ]
decDefs['I'] = _decodeBigInt

def _decodeVeryBigInt(read, n, typ, unpack=struct.unpack):
    if n == 1:
        return decode_long(read( unpack('I', read(4)) ))

    return [ decode_long(read( unpack('I', read(4))[0] )) for i in xrange(n) ]
decDefs['J'] = _decodeVeryBigInt


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


def _encodeFloat(x, pack=struct.pack):
    return 'd' + pack('d', x)
encDefs[float] = _encodeFloat

def _decodeFloat(read, n, typ, unpack=struct.unpack):
    if n == 1:
        return unpack('d', read(8))[0]

    return unpack(str(n) + 'd', read(n * 8))
decDefs['d'] = _decodeFloat

def _encodeComplex(x, pack=struct.pack):
    return 'C' + pack('2d', x.real, x.imag)
encDefs[complex] = _encodeComplex

def _decodeComplex(read, n, typ, unpack=struct.unpack):
    if n == 1:
        values = unpack('2d', read(16))
        return complex(*values)

    count = 2 * n # real, imag
    values = unpack(str(count) + 'd', read(8 * count))
    return [complex(*t) for t in zip(values[::2], values[1::2])]
decDefs['C'] = _decodeComplex

def _encodeStr(x):
    return _encodeCount(len(x)) + 'S' + x
encDefs[str] = _encodeStr

def _decodeStr(read, n, typ):
    return read(n)
decDefs['S'] = _decodeStr

def _encodeUnicode(x):
    x = x.encode('utf-8')
    return _encodeCount(len(x)) + 'U' + x
encDefs[unicode] = _encodeUnicode

def _decodeUnicode(read, n, typ):
    return unicode(read(n), 'utf-8')
decDefs['U'] = _decodeUnicode

def _encodeSlice(x, encDefs=encDefs):
    start = encDefs[type(x.start)](x.start)
    stop = encDefs[type(x.stop)](x.stop)
    step = encDefs[type(x.step)](x.step)

    return ':' + start + stop + step
encDefs[slice] = _encodeSlice

def _decodeSlice(read, n, typ, decDefs=decDefs):
    if n == 1:
        c = read(1); start = decDefs[c](read, 1, c)
        c = read(1); stop = decDefs[c](read, 1, c)
        c = read(1); step = decDefs[c](read, 1, c)
        return slice(start, stop, step)

    r = []
    for i in xrange(n):
        c = read(1); start = decDefs[c](read, 1, c)
        c = read(1); stop = decDefs[c](read, 1, c)
        c = read(1); step = decDefs[c](read, 1, c)
        r.append( slice(start, stop, step) )
    return r
decDefs[':'] = _decodeSlice



def _encodeCount(n, pack=struct.pack):
    if n == 1:
        return ''

    if n < 10:
        return str(n)

    if n >> 8: # doesn't fit into 1 byte
        return 'P' + pack('I', n)

    return 'p' + chr(n)


def _encodeContainer(x, visited=None, encDefs=encDefs, starts={tuple: '(', list: '[', set: '<', dict: '{'}, ends={tuple: ')', list: ']', set: '>', dict: '}'}, reducable='NTFbhiqIJdC:'):
    # handle recursion
    xId = id(x)
    if visited == None:
        visited = set()
    elif xId in visited:
        raise KeyError('recursive type')
    visited.add(xId)

    encodeCount = _encodeCount

    xTyp = type(x)
    r = [starts[xTyp]]

    if xTyp == set:
        x = sorted(x, key=type)
    elif xTyp == dict:
        keys = x.keys()
        keys.sort(key=type)
        x = it.chain(keys, [x[key] for key in keys])

    iX = iter(x)
    active = ''
    hasData = False
    count = 1

    for item in iX:
        ## reducable
        typ = type(item)
        if typ in starts: # is container type
            n = _encodeContainer(item, visited)
        else:
            n = encDefs[typ](item) # may raise key error

        first = n[0]
        if first == active:
            count += 1
            if hasData:
                data.append(n[1::])
            continue

        r.append(encodeCount(count) + active) # initial: '' + ''
        if hasData:
            r.extend(data)

        if first in reducable:
            active = first
            count = 1
            if len(n) != 1:
                hasData = True
                data = [n[1::]]
            else:
                hasData = False

            continue
        ## reducable

        ## not reducable
        r.append(n)

        for item in iX:
            typ = type(item)
            if typ in starts:
                r.append( _encodeContainer(item, visited) )
                continue

            n = encDefs[typ](item)

            first = n[0]
            if first not in reducable:
                r.append(n)
                continue

            active = first
            count = 1
            if len(n) != 1:
                hasData = True
                data = [n[1::]]
            else:
                hasData = False

            break
        else: # out of items; last item was not reducable
            break
        ## not reducable

    else: # last item was reducable
        r.append(encodeCount(count) + active)
        if hasData:
            r.extend(data)


    r.append(ends[xTyp])

    visited.remove(xId)
    return ''.join(r)
encDefs[tuple] = _encodeContainer
encDefs[list] = _encodeContainer
encDefs[set] = _encodeContainer
encDefs[dict] = _encodeContainer



class EndOfContainer(Exception):
    pass

class Interface(object):
    def __init__(self, atts):
        self.__dict__.update(atts)

def _decodeContainer(read, n, typ, endEx=EndOfContainer):
    if typ == '<': # set
        r = set()
        append = r.add
        extend = r.update
    else: # tuple, list or dict
        r = []
        append = r.append
        extend = r.extend

    containerTypes = [list, tuple]
    decodeHead = _decodeHead

    while True:
        n, c = decodeHead(read)

        try:
            item = decDefs[c](read, n, c)
        except endEx:
            break

        if n != 1 and type(item) in containerTypes:
            extend(item)
            continue

        append(item)

    if typ == '(': # tuple
        r = tuple(r)
    elif typ == '{': # dict
        r = dict(zip(r, r[len(r) / 2::]))

    return r
decDefs['('] = _decodeContainer
decDefs['['] = _decodeContainer
decDefs['<'] = _decodeContainer
decDefs['{'] = _decodeContainer

def _decodeEnd(read, n, typ, endEx=EndOfContainer):
    raise endEx
decDefs[')'] = _decodeEnd
decDefs[']'] = _decodeEnd
decDefs['>'] = _decodeEnd
decDefs['}'] = _decodeEnd



def _encodeBitarray(x):
    return _encodeCount(len(x)) + 'B' + x.tobytes()

def _decodeBitarray(read, n, typ):
    end = n >> 3 # l / 8
    pads = n & 0x7

    r = bitarray.bitarray()
    if pads:
        r.frombytes(read(end + 1))
        del r[-(8 - pads)::]

    else:
        r.frombytes(read(end))

    return r

if bitarray != None:
    encDefs[bitarray.bitarray] = _encodeBitarray
    decDefs['B'] = _decodeBitarray


def encode(x, encDefs=encDefs):
    try:
        return encDefs[type(x)](x)
    except KeyError:
        return None

def decode(file, decDefs=decDefs):
    read = file.read
    n, c = _decodeHead(read)
    return decDefs[c](read, n, c)


def _decodeHead(read, unpack=struct.unpack):
    '''
    '''
    c = read(1)

    if '0' <= c and c <= '9':
        n = ord(c) - 48 # - ord('0')
        c = read(1)
    elif c == 'p':
        n = ord(read(1))
        c = read(1)
    elif c == 'P':
        n, = unpack('I', read(4))
        c = read(1)
    else:
        n = 1

    return n, c

def decodes(s):
    f = StringIO.StringIO(s)
    return decode(f)



allEncs = encDefs.copy()
allDecs = decDefs.copy()

def enableType(typ):
    encDefs[typ] = allEncs[typ]

def enableTypes(typs):
    for typ in typs:
        enableType(typ)

def disableType(typ):
    try:
        del encDefs[typ]
    except KeyError: pass

def disableTypes(typs):
    for typ in typs:
        disableType(typ)

disableTypes([tuple, list, set, dict])



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
    import pickle
    import cPickle
    import sys
    import time

    enableTypes([tuple, list, set, dict])

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

            a = encode(n)
            if a == None:
                print 'Warning: is not primitive'
                continue
            b = cPickle.dumps(n, 2)

            print repr(a), '(%s)' % len(a)
            print float(len(a)) / len(b)




    def check(x, verbose=True):
        print x, '=',
        a = encode(x)

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
    check( {None: True, 1: 0.1} )
    check( (1, [2, set([3, tuple()]), set([])], []) )

    for value in True, False, 1, 2 ** 128, 2 ** (8 * 356), 3.14, complex(1.2, 3.4), slice(0, None, ()):
        check( [value] * 7 )
    check( {1: 2, 3: [None]} )

    a = []
    a.append(a)
    assert encode( a ) == None

    if bitarray != None:
        check( bitarray.bitarray([1, 0, 0, 1]))
        check( bitarray.bitarray([1, 0, 0, 1, 0, 0, 1, 0]))
        check( bitarray.bitarray([1, 0, 0, 1, 0, 0, 1, 0, 1, 1]))



    import random

    def compare(n, l):
        pSum = 0.

        for i in xrange(n):
            obj = tuple(randPrim(0, l))

            a = len(encode(obj))
            b = len(cPickle.dumps(obj, 2))

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
            b = len(cPickle.dumps(obj, 2))

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
            b = len(cPickle.dumps(obj, 2))

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
            print float(len(encode(a))) / len(cPickle.dumps(a, 2)) # 0.95 @ n = 8000



    n = 100
    l = 1000
    objs = [tuple(randPrim()) for i in xrange(l)]

    def timeit(f, x, n):
        begin = time.time()

        for i in xrange(n):
            for y in x:
                f(y)

        end = time.time()
        return end - begin

    print
    print 'random performance'

    t_cDumps = lambda x, f=cPickle.dumps: f(x, 2)
    t_encode = lambda x, f=encode: f(x)
    t_dumps = lambda x, f=pickle.dumps: f(x, 2)

    print 'cPickle.dumps:', timeit(t_cDumps, objs, n)
    print 'encode:', timeit(t_encode, objs, n)
    print 'pickle.dumps:', timeit(t_dumps, objs, n)

    dumps = [cPickle.dumps(obj, 2) for obj in objs]
    encs = [encode(obj) for obj in objs]

    print 'cPickle.loads:', timeit(cPickle.loads, dumps, n)
    print 'decode:', timeit(decodes, encs, n)
    print 'pickle.loads:', timeit(pickle.loads, dumps, n)


    n = 10 ** 2
    c = 10 ** 3



    print
    print 'single object performance'
    print 'encode'
    x = [None] * c
    print 'None', timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[bool]; x = [f(random) for i in xrange(c)]
    print 'bool', timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[1]; x = [f(random) for i in xrange(c)]
    print 1, timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[2]; x = [f(random) for i in xrange(c)]
    print 2, timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[3]; x = [f(random) for i in xrange(c)]
    print 3, timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[4]; x = [f(random) for i in xrange(c)]
    print 4, timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[5]; x = [f(random) for i in xrange(c)]
    print 5, timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[6]; x = [f(random) for i in xrange(c)]
    print 6, timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[7]; x = [f(random) for i in xrange(c)]
    print 7, timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[8]; x = [f(random) for i in xrange(c)]
    print 8, timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[16]; x = [f(random) for i in xrange(c)]
    print 16, timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[float]; x = [f(random) for i in xrange(c)]
    print 'float', timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)
    f = randGens[complex]; x = [f(random) for i in xrange(c)]
    print 'complex', timeit(t_cDumps, x, n), timeit(t_encode, x, n), timeit(t_dumps, x, n)

    print 'decode'
    x = [None] * c; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 'None', timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[bool]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 'bool', timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[1]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 1, timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[2]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 2, timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[3]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 3, timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[4]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 4, timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[5]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 5, timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[6]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 6, timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[7]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 7, timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[8]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 8, timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[16]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 16, timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[float]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 'float', timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)
    f = randGens[complex]; x = [f(random) for i in xrange(c)]; dumps = map(t_cDumps, x); encs = map(encode, x)
    print 'complex', timeit(cPickle.loads, dumps, n), timeit(decodes, encs, n), timeit(pickle.loads, dumps, n)




