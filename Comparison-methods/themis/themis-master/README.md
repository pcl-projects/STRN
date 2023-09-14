[![Build Status](https://travis-ci.org/atlassian/themis.png)](https://travis-ci.org/atlassian/themis)
[![Coverage Status](https://coveralls.io/repos/github/atlassian/themis/badge.svg?branch=master)](https://coveralls.io/github/atlassian/themis?branch=master)
[![PyPI Version](https://badge.fury.io/py/themis-autoscaler.svg)](https://badge.fury.io/py/themis-autoscaler)
[![PyPI License](https://img.shields.io/pypi/l/themis-autoscaler.svg)](https://img.shields.io/pypi/l/themis-autoscaler.svg)
[![Code Climate](https://codeclimate.com/github/atlassian/themis/badges/gpa.svg)](https://codeclimate.com/github/atlassian/themis)

# Themis - Autoscaling EMR Clusters and Kinesis Streams on AWS

Themis is a framework for autoscaling Cloud infrastructure on Amazon Web Services (AWS).

It currently supports autoscaling for the following services:
* Elastic Map Reduce (EMR) clusters
* Kinesis streams

Themis in Greek mythology is the goddess of divine order, justice, and fairness, just like
the Themis autoscaler whose job is to continuously maintain a tradeoff between costs and
performance.

![Themis](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Justitia%2C_Jost_Amman.png/200px-Justitia%2C_Jost_Amman.png)

## Background

Elastic Map Reduce (EMR) is a managed Hadoop environment for Big Data analytics offered by Amazon Web Services (AWS).
The architecture of EMR clusters is built for scalability, with a single master node and multiple worker nodes that
can be dynamically added and removed via Amazon's API.

Due to different usage patterns (e.g., high load during work hours, no load over night), the cluster may become either
underprovisioned (users experience bad performance) or overprovisioned (cluster is idle, causing a waste of resources
and unnecessary costs).

However, while autoscaling has become state-of-the-art for applications in Amazon's Elastic Compute Cloud (EC2),
currently there exists no out-of-the-box solution for autoscaling analytics clusters on EMR.

The "Themis" autoscaling tool actively monitors the performance of a user's EMR clusters and automatically scales the
cluster up and down where appropriate. Themis supports
two modes:
* **Reactive autoscaling**: Add and remove nodes based on the current load of the cluster
* **Proactive autoscaling**: Define minimum number of nodes based on a schedule (e.g., 10+ nodes during working hours, but only 2 nodes over night)

A Web user interface (UI) is available to display the key data. The rules for autoscaling can be customized in a configuration file or via the Web UI.

![Autoscaling Example](https://raw.githubusercontent.com/atlassian/themis/master/themis/web/img/scaling.png)

## Running in Docker

If you simply want to spin up a Themis instance in Docker, run the following command:

```
docker run -it -p 8080:8080 -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... atlassianlabs/themis
```

Make sure to set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables properly. The AWS region defaults to `us-east-1`, you can override it by passing in the `AWS_DEFAULT_REGION` environment variable.

## Requirements

* `make`
* `python`
* `pip` (python package manager)
* `npm` (node.js package manager, needed for Web UI dependencies)

## Installing

The simplest way to install the latest version of Themis is via `pip`:

```
pip install themis-autoscaler
```

## Developing

To install the tool dependencies for local development, run the following command:

```
make build
```

This will install the required pip dependencies in a local Python virtualenv directory 
(.venv), as well as the node modules for the Web UI in `./web/node_modules/`. 
Depending in your system, some pip/npm modules may require additional native libs installed.
Under Redhat/CentOS or Amazon Linux, you may first need to run the following command:

```
sudo yum -y install blas-devel lapack-devel numpy-f2py
```

Under Ubuntu, the following packages are required (tested under Ubuntu 15.04):

```
sudo apt-get -y install libblas-dev liblapack-dev python-numpy gfortran
```

## Testing

The project comes with a set of unit and integration tests which can be kicked off via a make
target:

```
make test
```

The test framework automatically collects and reports test coverage metrics (line coverage):

```
Name                         Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------
themis/config.py                            165     21    87%   217, 231-240, 242, 260-270
themis/model/aws_model.py                    46      3    93%   50, 74, 77
themis/model/emr_model.py                    25      3    88%   17-18, 22
themis/model/kinesis_model.py                65     27    58%   15-17, 20-22, 25, 46-48, 54-55, 68-73, 76-77
themis/model/resources_model.py              17      4    76%   21-24
themis/monitoring/database.py                44     30    32%   15-19, 23-27, 29-37, 41-47, 52-60
themis/monitoring/kinesis_monitoring.py     119     31    74%   18-30, 41, 59-60, 64, 66-67, 111-115, 142
themis/monitoring/resources.py               45     17    62%   23-24, 30, 35-39, 46, 51-61
themis/scaling/emr_scaling.py               173     57    67%   27, 145, 150, 162-170, 174-175, 179-188
themis/scaling/kinesis_scaling.py            90     43    52%   14-15, 19, 23, 25-28, 30-39, 44, 62-66, 68-69
...
-----------------------------------------------------------------------
TOTAL                                      1833    367    80%   
----------------------------------------------------------------------
Ran 13 tests in 16.522s

OK
```

We seek to further improve the test coverage as the framework evolves.

## Configuration

Themis autoscaler relies on the `aws` command line interface (AWS CLI) to be installed and
configured with your AWS account credentials. If you have not yet done so, run this command and
follow the instructions:

```
aws configure
```

For the configuration of the autoscaler itself, there are a number of settings that you can
configure directly in the Web UI.

### Global configuration settings

Configuration Key					| 	Description
------------------------------------|--------------------------------
`autoscaling_clusters`				|	Comma-separated list of cluster IDs to auto-scale
`loop_interval_secs`				|	Loop interval seconds
`monitoring_interval_secs`			|	Time period (seconds) of historical monitoring data to consider for scaling decisions
`ssh_keys`							|	Comma-separated list of SSH public key files to use for connecting to the clusters.


### EMR configuration (per cluster)

Configuration Key					| 	Description
------------------------------------|--------------------------------
`baseline_nodes`					|	Number of baseline nodes to use for comparing costs and calculating savings
`custom_domain_name`				|	Custom domain name to apply to all nodes in cluster (override aws-cli result)
`downscale_expr`					|	Trigger cluster downscaling by the number of nodes this expression evaluates to
`monitoring_interval_secs`			|	Time period (seconds) of historical monitoring data to consider for scaling decisions
`group_or_preferred_market`			|	Comma separated list of task instance groups and/or instance markets to increase/decrease depending on order, e.g., "ig-12345,SPOT,ON_DEMAND" means to autoscale task group ig-12345 if available, otherwise any SPOT group, or if necessary ON_DEMAND groups
`time_based_scaling`				|	A JSON string that maps date regular expressions to minimum number of nodes. Dates to match against are formatted as "%a %Y-%m-%d %H:%M:%S". Example config: {"(Mon|Tue|Wed|Thu|Fri).*01:.*:.*": 1}
`upscale_expr`						|	Trigger cluster upscaling by the number of nodes this expression evaluates to

### Kinesis configuration (per stream)

Configuration Key					| 	Description
------------------------------------|--------------------------------
`enable_enhanced_monitoring`		|	Enable enhanced monitoring. A value of "true" enables per-shard monitoring with ShardLevelMetrics=ALL
`stream_upscale_expr`				|	Trigger stream upscaling by the number of shards this expression evaluates to
`stream_downscale_expr`				|	Trigger stream downscaling by the number of shards this expression evaluates to

## Running

The Makefile contains a target to conveniently run the server application. Prior to that, make
sure that ssh agent forwarding is enabled in the shell that executes the server.

```
# enable ssh agent forwarding
eval `ssh-agent -s`
# start the server on port 9090
make server
```

## Change Log

* v0.2.9: Add system load to monitoring metrics for EMR clusters
* v0.2.8: Fix shard-level metrics for Kinesis, add min/max/avg to UI
* v0.2.7: Extended monitoring metrics for Kinesis
* v0.2.6: Fix merging of adjacent Kinesis shards. Fix duplicates in EMR downscale candidates
* v0.2.5: Add SQLAlchemy and store config and monitoring data in a proper database
* v0.2.4: Support temporary STS tokens via assumed IAM roles
* v0.2.3: Use boto3 for AWS API calls
* v0.2.1: Add Dockerfile, publish image to Docker hub
* v0.2.0: Kinesis autoscaling; some fixes in UI; extend tests; update documentation
* v0.1.15: Initial version of auto-scaling Kinesis streams; first release tag
* v0.1.14: Add DSL support for max,min,sum of CPU and RAM in EMR autoscaling
* v0.1.12: Add DSL support for preferred instance market config; allow to configure specific task group
* v0.1.9: Add coveralls badge to README
* v0.1.8: Rework code, add model classes, make model and configuration generic, prepare for adding Kinesis autoscaling
* v0.1.7: Add support for custom domain names
* v0.1.4: First functional package published on PyPi
* v0.1.0: Initial release

## Contributors

The following developers have actively contributed to Themis (in order of appearance of their first contribution):

* [whummer](https://github.com/whummer) (Waldemar Hummer)
* [FJK-NZ](https://github.com/FJK-NZ) (Feliks Krawczyk)
* [hye1](https://github.com/hye1) (Hao Ye)
* [tdawber](https://github.com/tdawber) (Thomas Dawber)
* [patanachai](https://github.com/patanachai) (Patanachai Tangchaisin)

## Contributing

We welcome feedback, bug reports, and pull requests!

For pull requests, please stick to the following guidelines:

* Add tests for any new features and bug fixes. Ideally, each PR should increase the test coverage.
* Follow the existing style (e.g., tab indents).
* Put a reasonable amount of comments into the code.
* Separate unrelated changes into multiple pull requests.

Please note that we need to collect a signed **Contributors License Agreement** from each
individual developer who contributes code to this repository. Please refer to the following links:

* [https://developer.atlassian.com/opensource/](https://developer.atlassian.com/opensource/)
* [https://na2.docusign.net/Member/PowerFormSigning.aspx?PowerFormId=3f94fbdc-2fbe-46ac-b14c-5d152700ae5d](https://na2.docusign.net/Member/PowerFormSigning.aspx?PowerFormId=3f94fbdc-2fbe-46ac-b14c-5d152700ae5d)

## License

Copyright (c) 2016 Atlassian and others.

Themis is released under the Apache License, Version 2.0 (see LICENSE.txt).

We build on a number of third-party software tools, with the following licenses:

Third-Party software		| 	License
----------------------------|-----------------------
**Python/pip modules:**		|
awscli						|	Apache License 2.0
coverage 					|	Apache License 2.0
cssselect					|	BSD License
docopt						|	MIT License
flask						|	BSD License
flask-swagger				|	MIT License
nose						|	GNU LGPL
pyhive						|	Apache License 2.0
scipy						|	BSD License
**Node.js/npm modules:**	|
swagger-client				|	Apache License 2.0
jquery						|	MIT License
angular-tablesort			|	MIT License
almond						|	MIT License
angular						|	MIT License
angular-ui-router			|	MIT License
angular-resource			|	MIT License
angular-sanitize			|	MIT License
showdown					|	BSD 3-Clause License
angular-aui					|	Atlassian Design Guidelines License
