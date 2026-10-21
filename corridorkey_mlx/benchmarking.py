# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""Optional macOS physical-footprint sampling for the CLI's benchmark report."""
import ctypes
import os
import threading


class RUsage(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in
        ('user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins',
         'wired_size', 'resident_size', 'phys_footprint', 'proc_start_abstime', 'proc_exit_abstime')]


class Footprint:
    """Record process physical footprint, including MLX's unified-memory use."""
    def __init__(self):
        self.lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.proc_pid_rusage.restype = ctypes.c_int
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.peak = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            usage = RUsage()
            if self.lib.proc_pid_rusage(os.getpid(), 0, ctypes.byref(usage)) == 0:
                with self.lock:
                    self.peak = max(self.peak, usage.phys_footprint)
            self.stop_event.wait(.02)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
