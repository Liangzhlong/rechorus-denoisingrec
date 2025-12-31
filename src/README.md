# Source Code

`main.py` serves as the entrance of our framework, and there are three main packages. 

### Structure

- `helpers\`
  - `BaseReader.py`: read dataset csv into DataFrame and append necessary information (e.g. interaction history)
  - `ContextReader.py`: inherited from BaseReader, read user&item metadata, and count statistics about all context features
  - `ContextSeqReader.py`: inherited from ContextReader, append interaction history with situation context features.
  - `ImpressionReader.py`: inherited from BaseReader, group interactions with the same impression id into an instance. 
  - `BaseRunner.py`: control the training and evaluation process of a model
  - `CTRRunner.py`: inherited from BaseRunner, train and evaluate a model with binary label. (Click-through-rate Predition task)
  - `ImpressionRunner.py`: inherited from BaseRunner, train and evaluate a model with impression-based logs (Variable lengths of positive and negative items in a list).
  - `...`: customize helpers with specific functions
- `models\`
  - `BaseModel.py`: basic model classes and dataset classes, with some common functions of a model
  - `BaseContextModel.py`: inherited from BaseModel, add context features for base model
  - `BaseImpressionModel.py`: inherited from BaseModel, construct data batch in impressions
  - `...`: customize models inherited from classes in *BaseModel*
- `utils\`
  - `layers.py`: common modules for model definition (e.g. attention and MLP blocks)
  - `utils.py`: some utils functions
- `main.py`: main entrance, connect all the modules
- `exp.py`: repeat experiments in *run.sh* and save averaged results to csv 

### Define a New Model

Generally we can define a new class inheriting *GeneralModel* (a subclass of *BaseModel*), as well as the inner class *Dataset*. The following functions need to be implement at least:

```python
class NewModel(GeneralModel):
    reader = 'BaseReader'  # assign a reader class, BaseReader by default
    runner = 'BaseRunner'  # assign a runner class, BaseRunner by default

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        # define parameters in the model

    def forward(self, feed_dict):
        # generate prediction (ranking score according to tensors in feed_dict)
        item_id = feed_dict['item_id']  # [batch_size, -1]
        user_id = feed_dict['user_id']  # [batch_size]
        prediction = (...)
        out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1)}
        return out_dict

    class Dataset(GeneralModel.Dataset):
        # construct feed_dict for a single instance (called by __getitem__)
        # will be collated to a integrated feed dict for each batch
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            (...)
            return feed_dict
```

If the model definition is more complicated, you can inherit other functions in *BaseModel* (e.g. `loss`, `customize_parameters`) and *Dataset* (e.g. `_prepare`, `actions_before_epoch`), which needs deeper understandings about [BaseModel.py](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BaseModel.py) and [BaseRunner.py](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseRunner.py). You can also implement a new runner class to accommodate different experimental settings.

### Running DenoisingRec Models (e.g., T_CE)

We have integrated models from [DenoisingRec](https://github.com/Wenhui-Yu/DenoisingRec) (e.g., T_CE) into this framework.

To run **T_CE** (Truncated Cross Entropy) on the `amazon_book` dataset:

1.  **Prepare Data**: Ensure the dataset files (`amazon_book.train.rating`, `amazon_book.valid.rating`, `amazon_book.test.negative`) are located in `data/amazon_book/`.
2.  **Run Command**:

```bash
python main.py --model_name T_CE --dataset amazon_book --model_type NeuMF-end --drop_rate 0.2 --num_gradual 30000 --test_all 1
```

**Key Arguments for T_CE:**

*   `--model_name T_CE`: Use the T_CE model.
*   `--model_type`: Backbone model type (`MLP`, `GMF`, `NeuMF-end`). Default is `NeuMF-end`.
*   `--drop_rate`: The final drop rate for the truncated loss. Default `0.2`.
*   `--num_gradual`: Number of iterations for the drop rate to increase linearly. Default `30000`.
*   `--exponent`: Exponent for drop rate schedule. Default `1` (linear).
*   `--test_all 1`: Enable full-ranking evaluation (ranking against all non-interacted items), consistent with DenoisingRec.
*   `--regenerate 1`: If you have modified the data reader or need to reload raw data, use this flag to ignore cached `.pkl` files.

**Note**: The `DenoisingReader` is used automatically for `T_CE` to handle the specific data format of DenoisingRec datasets.

