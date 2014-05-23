encprim
=======

is a <b>serializer</b> for primitive python objects, similar to json or msgpack but written in pure python

Overview
--------

1. initially designed as extension for the python module <b>struct</b>
2. later inspired by msgpack and rewritten
3. encodes <b>None, bool, int/long (different sizes), float, complex, str, slice</b> and <b>bitarray</b>
4. supports nesting in <b>tuple, list, set</b> and <b>dict</b>
5. detects and exploits type repetitions

Output Syntax
-------------

<i>value:</i> [count]type[data]<br />
<i>tuple / list / set:</i> sequence of values enclosed by () / [] / <><br />
<i>dict:</i> sequence of value pairs (key, value) enclosed by {}<br />

Examples with data replaced by "."<br />
(2TF)  ...  (True, True, False)<br />
{i.d.N(2C....)}  ...  {1: 5.1, None: (3.0-2.0i, 1.0+4.0i)}<br />

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

- why not use pickle:

pickle is great, especially its power to serialize really everything there is, including functions/classes defined at \__main__ level thanks to Oren Tirosh's monkey patch*.
But if you find yourself in the situation where you have to serialize a large number of small objects, then every byte may count.

- why not use existing serializers like json, msgpack

Besides the external dependency they exist for a number of other reasons and are therefore not 100% compliant with python types. For example**  json's dictionary keys need to be strings and msgpack can't distinct between tuple and list.

- why not

I found that pickle is quite efficient but for one exception: complex numbers. The output shouldn't be much larger than two doubles, but it somehow is. Also pickle seems to store a lot of unnecessary information when it is fed with a bitarray. This type is not built in, but I love it and use it extensively.

So I decided to put some effort into this module. I hope you like it or find your usecase.



\* http://code.activestate.com/recipes/572213-pickle-the-interactive-interpreter-state/<br />
** in my best knowledge
