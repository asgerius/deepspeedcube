from __future__ import annotations

from pelutils import log, TT
from pelutils.parser import Parser, Option

from deepspeedcube import HardwareInfo
from deepspeedcube.eval.eval import eval


options = (
    Option("max-depth",        default=30, type=int),
    Option("states-per-depth", default=100, type=int),
    Option("solver",           default="AStar"),
    Option("max-time",         default=1, type=float),
    Option("astar-lambda",     default=1, type=float),
    Option("astar-N",          default=100),
    Option("astar-d",          default=1),
)

if __name__ == "__main__":
    parser = Parser(*options, multiple_jobs=True)
    jobs = parser.parse_args(clear_folders=True)

    for i, job in enumerate(jobs):
        TT.reset()
        log.configure(f"{job.location}/train.log")
        with log.log_errors:
            log.section(f"Evaluating {i+1} / {len(jobs)}: {job.name}")
            log.log_repo()
            log(
                "Hardware info:",
                "CPU:     %s" % HardwareInfo.cpu,
                "Sockets: %i" % HardwareInfo.sockets,
                "Cores:   %i" % HardwareInfo.cores,
                "GPU:     %s" % HardwareInfo.gpu,
                sep="\n    ",
            )
            eval(job)
            log("Time distribution", TT)
