# NYT Data Science Interview Assignment

## Data Description

In `homework.json.gz` you will find a series of JSON objects
  (one object per decompressed line)
containing partial content of a number of New York Times assets
  (articles, feature pieces, etc.).
You will also find a set of opaque tags for each asset.
These tags correspond to the emotional reaction(s) to an asset,
as labeled by a human. Specifically:

|Column|Description|
|------|-----------|
|`headline`|The complete headline of the asset at time of first publication|
|`summary`|A sentence from the body of the article deemed representative of its overall content|
|`worker_id`|A numeric ID indicating the specific human who generated the emotional tags|
|`emotion_{0-9}`|Binary flags indicating emotions evoked in the reader by the article headline and summary|


- Please construct a program that, when run, will do the following:

* Construct a predictive model of which emotional reactions are present in a given asset
* Output, to stderr or a file, an appropriate quantitative estimate of the model's ability to correctly predict emotional reactions
* Start a webserver that will predict emotional tags of asset data it receives over HTTP. Specifically:

```
curl -d '{ "headline": "headline_content", "summary": "summary_content", "worker_id": 1 }' http://localhost:8080
```

Should return data of the schema:
```
{ "emotion_0": 0.0, "emotion_1": 0.12, ... "emotion_9": 0.99 }
```

- Please complete this in Python; you may use any modules, libaries, or framew    orks you require.

- Please provide a brief writeup of your approach and decision making process regarding subjective design choices you made during the analysis, along with instructions for executing the program if necessary.

- Please feel free to contact us if you have any questions concerning the project.

- Please submit solutions as pushed+committed changes; please ask any questions as github issues.

Best regards, The Data Science Group, The New York Times
