encprim
=======

is a <b>serializer</b> for primitive python objects, similar to cPickle, json or msgpack but written in pure python. Find the reasons in the corresponding section below.

Overview
--------

1. initially designed as extension for the python module <b>struct</b>
2. later inspired by msgpack and pickle
3. encodes <b>None, bool, int/long (arbitrary size), float, complex, str, unicode, slice</b> and <b>bitarray</b>
4. supports nesting in <b>tuple, list, set</b> and <b>dict</b>
5. detects and exploits type repetitions

Output Syntax
-------------

<i>object:</i> [count]type[data]<br />
<i>tuple / list / set:</i> sequence of objects enclosed by () / [] / <><br />
<i>dict:</i> sequence of objects, keys first then values, enclosed by {}<br />

Examples with data replaced by "."<br />
(2TF2N)  ==  (True, True, False, None, None)<br />
{(2i..)()d.4s.}  ==  {(3, 7): 1.2, (): 'text'}<br />

Getting Started
---------------

Copy the encprim directory somewhere python will find it

> import encprim<br />
x = 1<br />
a = encprim.encode( x )<br />
print repr(a)<br />
b = encprim.decode( a )<br />
assert b == x<br />
<br />
import bitarray<br />
print encprim.encode( bitarray.bitarray('11011') )

encode returns None when the object contains non encodable types or if recursions exist.

You can start the test suite by executing the \__init__.py file directly. \__init__.out contains an example output.<br />
Add the argument "-i" to get into interactive mode, where you can type in python structs for which the encoded value is printed, alongside with its size in bytes and the size ratio compared to pickle (lower is better).

Reasons
-------

- why not use pickle

pickle is great, especially its power to serialize really everything there is, including functions/classes defined at \__main__ level thanks to Oren Tirosh's monkey patch*.
But if you find yourself in the situation where you have to serialize a large number of small objects, then every byte may count.

- why not use existing serializers like json, msgpack

Besides the external dependency they exist for a number of other reasons and are therefore not 100% compliant with python types. For example**  json's dictionary keys need to be strings and msgpack can't distinct between tuple and list.

- why not

I found that pickle is quite efficient but for one exception: complex numbers. The output shouldn't be much larger than two doubles, but it somehow is. Also pickle seems to store a lot of unnecessary information when it is fed with a bitarray. This type is not built in, but I love it and use it extensively.

So I decided to put some effort into this module. I hope you like it or find your usecase.

Performance
-----------

The test suite shows that encprim produces outputs which are, on average, 40% smaller compared to pickle. Depending on the data type the rate could rise above 90% (bitarrays with len < 16) or drop below significance (big integers). By design the best results are achieved with single values or a flat collection of same typed values.

Regarding runtime, the size optimizations come at a prize. Compared to pickle, which is a valid baseline because it is pure python too, encprim is, on average, about twice as fast when encoding and only a few percent faster when decoding. However, en/decoding single values is about 5 times faster than pickle. cPickle is fastest in any case but encprim comes very close when en/decoding single values :)


\* http://code.activestate.com/recipes/572213-pickle-the-interactive-interpreter-state/<br />
** in my best knowledge
