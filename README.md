### Running DenoisingRec Models (e.g., T_CE)

We have integrated models from [DenoisingRec](https://github.com/Wenhui-Yu/DenoisingRec) (e.g., T_CE) into this framework.

To run **T_CE** (Truncated Cross Entropy) on the `amazon_book` dataset:

1.  **Prepare Data**: Ensure the dataset files (amazon_book.train.rating, amazon_book.valid.rating, amazon_book.test.negative) are located in `data/amazon_book/`
2.  **Run Command**:

```bash
python main.py --model_name T_CE --dataset amazon_book --model_type NeuMF-end --drop_rate 0.2 --lr 0.0001 --num_gradual 30000 --test_all 1
```

**Key Arguments for T_CE:**

*   `--model_name T_CE`: Use the T_CE model.
*   `--model_type`: Backbone model type (MLP, GMF, NeuMF-end). Default is `NeuMF-end`
*   `--drop_rate`: The final drop rate for the truncated loss. Default is `0.2`
*   `--num_gradual`: Number of iterations for the drop rate to increase linearly. Default is `30000`
*   `--test_all 1`: Enable full-ranking evaluation (ranking against all non-interacted items), consistent with DenoisingRec.
*   `--lr`: learning-rate of modle. Default is `0.0001`

**Key Arguments for R_CE:**
*   `--alpha`:  the wight of the reweighted loss. Default is `0.2`

**Note**: The `DenoisingReader` is used automatically for `T_CE` to handle the specific data format of DenoisingRec datasets.

