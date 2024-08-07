# recommender

Change the path to the data!

## Running locally 

Start the SuperLink:

```bash
flower-superlink --insecure
```

Start the SuperNodes:

```bash
flower-client-app client_app:app --insecure --superlink 127.0.0.1:9092
flower-client-app client_app1:app --insecure --superlink 127.0.0.1:9092
```

Start the SuperExec:

```bash
flower-server-app server_app:app --superlink 127.0.0.1:9091
```

