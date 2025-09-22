import datetime, time, os, logging
import torch
logger = logging.getLogger(__name__)

def profiler(enable_profile: bool = False, profile_dir: str = 'tracing', global_step: int = 0):
    # get user defined profiler settings
    if enable_profile:
        dump_dir = './profiling'
        now=datetime.datetime.now()
        save_trace_dir = profile_dir
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        profile_freq = 5
        rank = torch.distributed.get_rank()
        # def trace_handler(prof):
        #     curr_trace_dir_name = "iteration_" + str(prof.step_num)
        #     curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
        #     if not os.path.exists(curr_trace_dir):
        #         os.makedirs(curr_trace_dir, exist_ok=True)
        #     logger.info(f"Dumping profiler traces at step {prof.step_num}")
        #     begin = time.monotonic()
        #     prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")
        #     logger.info(
        #         f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
        #     )
        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)
        warmup, active = 3, 1
        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{trace_dir}'),
            record_shapes=True,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            return torch_profiler
    else:
        return None
