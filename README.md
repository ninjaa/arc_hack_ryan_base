# ARC Draw more samples

## Forked from https://github.com/rgreenblatt/arc_draw_more_samples_pub

## Presentation

[Presentation](https://docs.google.com/presentation/d/1TSviK0q04J6OOje3KKVQHnA0xuPeSKWs7TfWzkFDKc8/edit)
[Notes](https://docs.google.com/document/d/16cyM-GyRWkhQ7KGeLOaCaGW4t9OtOqRF1IXrzi8AGFA/edit)

### Steps, Experiments & Results

- Recreated Ryan's environment locally
- Studied how it works, see [notes](https://docs.google.com/document/d/16cyM-GyRWkhQ7KGeLOaCaGW4t9OtOqRF1IXrzi8AGFA/edit)
- Attempted dimensions hint injection, results in [presentation](https://docs.google.com/presentation/d/1TSviK0q04J6OOje3KKVQHnA0xuPeSKWs7TfWzkFDKc8/edit)
- Attempted GPT-4 substitution, it did not work
- Created arckit-analysis and saved results in [results.json](./arckit-analysis/results.json)
- Did not have time to inject arckit-analysis results to test if they improved performance
- Shaw also created GPTCoder the next day https://github.com/lalalune/gptcoder & is trying that now



### Contributors

Aditya Advani, Shaw Walters, Vincent Wu, Ilia Zintchenko


## Original README

See [my blog post on substack for details](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt).

The main script is arc_solve/solve.py. This requires a redis server running at port "6381" (for caching), several hours of time, a considerable amount of memory, an openai key, and a bunch of money.

Required dependencies are tqdm, numpy, scipy, skimage, attrs, cattrs, nest_asyncio, redis-py, matplotlib, anthropic (not actually needed), and openai==0.28.1. (I might make an actual requirements.txt file later.)

Data can be loaded from jsons which can be found [at this drive link](https://drive.google.com/file/d/1t3LmW0oxnRHTksgeUrMwPYZMZ8dOb4X4/view?usp=sharing) and visualized and viewed using arc_solve/load_and_viz.py. (I may add additional plots and display later.)
