"""Windows current-user DPAPI storage. Plaintext keys never go to files or logs."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
from .clock import CanonicalError
from .provider import FootballData

class _Blob(ctypes.Structure):
    _fields_=[("size",wintypes.DWORD),("data",ctypes.POINTER(ctypes.c_ubyte))]


def key_path(database):
    return Path(database).with_suffix(".key.dpapi")


def _crypt(raw,decrypt=False):
    if os.name!="nt":raise CanonicalError("CURRENT_USER_DPAPI_REQUIRES_WINDOWS")
    crypt=ctypes.WinDLL("crypt32",use_last_error=True)
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    buffer=(ctypes.c_ubyte*len(raw)).from_buffer_copy(raw)
    source=_Blob(len(raw),buffer);target=_Blob()
    kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    try:
        if decrypt:
            fn=crypt.CryptUnprotectData
            fn.argtypes=[ctypes.POINTER(_Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(_Blob)]
            success=fn(ctypes.byref(source),None,None,None,None,1,ctypes.byref(target))
        else:
            fn=crypt.CryptProtectData
            fn.argtypes=[ctypes.POINTER(_Blob),wintypes.LPCWSTR,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(_Blob)]
            success=fn(ctypes.byref(source),"Betting Partner football-data credential",None,None,None,1,ctypes.byref(target))
        if not success:raise CanonicalError("CURRENT_USER_DPAPI_FAILED")
        return ctypes.string_at(target.data,target.size)
    finally:
        ctypes.memset(buffer,0,len(buffer))
        if target.data:
            ctypes.memset(target.data,0,target.size)
            kernel.LocalFree(ctypes.cast(target.data,ctypes.c_void_p))


def save_key(database,key):
    FootballData(key)  # Validate only; no network.
    encrypted=_crypt(key.encode("ascii"))
    path=key_path(database);path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create: rotation must be deliberate, never an unnoticed overwrite.
    with path.open("xb") as stream:
        stream.write(encrypted);stream.flush();os.fsync(stream.fileno())
    return {"state":"KEY_PROTECTED_FOR_CURRENT_WINDOWS_USER","path":str(path.resolve()),
            "portable_backup":False,"plaintext_written":False}


def load_key(database):
    path=key_path(database)
    with path.open("rb") as stream:encrypted=stream.read(16385)
    if not encrypted or len(encrypted)>16384:raise CanonicalError("PROTECTED_KEY_INVALID")
    try:key=_crypt(encrypted,True).decode("ascii")
    except UnicodeError as exc:raise CanonicalError("PROTECTED_KEY_INVALID") from exc
    FootballData(key)
    return key

