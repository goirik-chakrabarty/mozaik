import os, sys, runpy, cProfile, atexit
os.chdir('/project')
sys.path.insert(0, '/project')          # so run.py's `from experiments import ...` resolves
sys.argv = ['run.py','nest','12','param/defaults','lgn_stepcurrentsource_noise_seed','999989','perf_prof']
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('PMI_RANK','0')))
if rank == 0:
    pr = cProfile.Profile(); pr.enable()
    def _dump():
        pr.disable(); pr.dump_stats('/scr/getdata_rank0.prof'); sys.stderr.write('PROFILE DUMPED\n')
    atexit.register(_dump)
runpy.run_path('/project/run.py', run_name='__main__')
