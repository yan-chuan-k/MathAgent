# V3.1.1 Python Compatibility Hotfix

The V3.1.0 live-evaluation hardening added a union type annotation to
`intern_s1_client.py`.  That file did not use postponed annotation evaluation,
so Python versions before 3.10 raised a `TypeError` while importing the module:

```text
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

The annotation now uses the already-imported `Optional` type and is compatible
with the Python versions supported by the rest of the project.  No API key is
stored in the source tree or release archive.
