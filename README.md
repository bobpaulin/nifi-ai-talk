# nifi-ai-talk

## Setup

Download Apache NiFi 2.0.0-M4 or greater

```
wget https://dlcdn.apache.org/nifi/2.0.0-M4/nifi-2.0.0-M4-bin.zip
```

Unzip Apache NiFi

```
unzip nifi-2.0.0-M4-bin.zip
```

Uncomment Python Commands in conf/nifi.properties

```
#####################
# Python Extensions #
#####################
# Uncomment in order to enable Python Extensions.
nifi.python.command=python3
```

Set Password

```
cd nifi-2.0.0-M4
bin/nifi.sh set-single-user-credentials admin administrator12
```

Start Apache NiFi

```
bin/nifi.sh start
```

Open Browser and Login with:

username: admin

password: administrator12


Create a new Process Groups

Import from flows from 

flow_defs/InitElastic.json

flow_defs/NifiJava.json

flow_defs/TestTable.json

Build table_detection_processor NAR

```
cd nifi-ai-talk/table-detection-processor
hatch build -t nar
```

Install table_detection_processor NAR

```
 cp ../git/nifi-ai-talk/table-detection-processor/dist/table_detection_processor-0.0.1.nar extensions
```

Wait for Flow to startup and procesor to install.  

Copy chart file to /Users/bpaulin/table-img-in (or change this in the GetFile Processor

```
git/nifi-ai-talk/apple-chart-100px-whitespace.png /Users/bpaulin/table-img-in
```
