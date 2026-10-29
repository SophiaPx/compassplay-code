# CompassPlay integration with AZR and verl.
import os


_APPLIED = False


def _rocm_get_device_uuid(cls, device_id: int = 0) -> str:
    physical_id = cls.device_id_to_physical_device_id(device_id)
    try:
        import amdsmi

        try:
            amdsmi.amdsmi_init()
        except Exception:
            pass
        handles = amdsmi.amdsmi_get_processor_handles()
        return str(amdsmi.amdsmi_get_gpu_device_uuid(handles[physical_id]))
    except Exception:
        import socket

        return f"ROCM-{socket.gethostname()}-{physical_id}"


def apply() -> str:
    global _APPLIED
    if _APPLIED:
        return "[VLLM_ROCM_PATCH] already applied"

    try:
        from vllm.platforms import current_platform
    except Exception as e:
        return f"[VLLM_ROCM_PATCH] skipped (import vllm.platforms failed: {type(e).__name__}: {e})"

    platform_cls = type(current_platform)
    try:
        uuid = current_platform.get_device_uuid(0)
    except NotImplementedError:
        platform_cls.get_device_uuid = classmethod(_rocm_get_device_uuid)
        _APPLIED = True
        try:
            uuid = current_platform.get_device_uuid(0)
        except Exception as e:
            return f"[VLLM_ROCM_PATCH] patched but lookup failed: {type(e).__name__}: {e}"
        return f"[VLLM_ROCM_PATCH] added get_device_uuid to {platform_cls.__name__}, device0={uuid}"
    except Exception as e:
        return f"[VLLM_ROCM_PATCH] native implementation raised (not replaced): {type(e).__name__}: {e}"

    _APPLIED = True
    return f"[VLLM_ROCM_PATCH] {platform_cls.__name__} implements get_device_uuid natively, device0={uuid}"


def apply_and_log() -> None:
    msg = apply()
    print(msg, flush=True)
    if os.environ.get("VERL_LOGGING_LEVEL", "").upper() == "DEBUG":
        print(f"[VLLM_ROCM_PATCH] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}", flush=True)
