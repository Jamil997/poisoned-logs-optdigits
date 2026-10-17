"""Run the frozen audit unchanged, caching compressed arrays once in memory.

Added after the confirmation run to eliminate repeated ZIP decompression.
It changes I/O only; audit.py and all prespecified computation remain unchanged.
"""
import audit

original_load=audit.np.load

class MemoryArchive(dict):
    @property
    def files(self):return list(self)

def eager_load(*args,**kwargs):
    result=original_load(*args,**kwargs)
    if hasattr(result,'files'):
        with result:
            return MemoryArchive({key:result[key] for key in result.files})
    return result

if __name__=='__main__':
    audit.np.load=eager_load
    audit.main(audit.ROOT/'results')
