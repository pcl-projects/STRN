from flask import Flask, request, make_response
from flask_swagger import swagger
import json
import time
import threading


def serve(port):

    # mocking ganglia requests:
    # ?h=%s&cs=%s&ce=%s&c=%s&g=%s&json=1' % (cluster_ip, host, start_time, end_time, cluster_id, type_param)
    class GangliaApp(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.cpu = None
            self.mem = None
            self.sysload = None
            self.app = Flask('testapp')

        def run(self):
            @self.app.route('/ganglia/graph.php')
            def graph():
                graph_type = request.args.get('g')
                cpu = self.cpu if graph_type == 'cpu_report' else None
                mem = self.mem if graph_type == 'mem_report' else None
                sysload = self.sysload if graph_type == 'load_report' else None
                result = mock_ganglia(cpu=cpu, mem=mem, sysload=sysload)
                return result
            self.app.run(port=int(port), host='0.0.0.0')
    app = GangliaApp()
    app.daemon = True
    app.start()
    time.sleep(1)
    return app


def mock_ganglia(cpu=None, mem=None, sysload=None):
    from themis.util import common

    result = []
    cpu_idle_dp = []
    mem_total_dp = []
    mem_free_dp = []
    sysload_dp = []
    cpu_idle = {
        "ds_name": "cpu_idle", "cluster_name": "", "graph_type": "stack", "host_name": "",
        "metric_name": "Idle\\g", "color": "#e2e2f2", "datapoints": cpu_idle_dp
    }
    mem_total = {
        "ds_name": "bmem_total", "cluster_name": "", "graph_type": "line", "host_name": "",
        "metric_name": "Total\\g", "color": "#FF0000", "datapoints": mem_total_dp
    }
    mem_free = {
        "ds_name": "bmem_free", "cluster_name": "", "graph_type": "stack", "host_name": "",
        "metric_name": "Free\\g", "color": "#f0ffc0", "datapoints": mem_free_dp
    }
    sysload_total = {
        "ds_name": "a0", "cluster_name": "", "graph_type": "stack", "host_name": "",
        "metric_name": "1-min", "color": "#BBBBBB", "datapoints": sysload_dp
    }
    result.append(cpu_idle)
    result.append(mem_total)
    result.append(mem_free)
    result.append(sysload_total)
    end = common.now()
    start = end - 10 * 60
    t = start
    while t < end:
        if cpu:
            cpu_value = "NaN" if (t - start) < 60 * 2 else 100.0 - cpu
            cpu_idle_dp.append([cpu_value, t])
        if mem:
            mem_total = "NaN" if (t - start) < 60 * 2 else 64429572096.0
            mem_total_dp.append([mem_total, t])
            mem_free = "NaN" if (t - start) < 60 * 2 else mem_total * (mem / 100.0)
            mem_free_dp.append([mem_free, t])
        if sysload:
            load_value = "NaN" if (t - start) < 60 * 2 else sysload
            sysload_dp.append([load_value, t])
        t += 15

    return make_response(json.dumps(result))
