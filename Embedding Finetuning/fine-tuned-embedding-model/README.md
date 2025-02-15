---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:28
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: 'AMZN Balance Sheet as of 2024-09-30:

    - Total Assets: 584,626,000,000

    - Treasury Shares Number: 515000000.0

    - Ordinary Shares Number: 10511000000.0

    - Share Issued: 11026000000.0

    - Net Debt: nan

    - Total Debt: 134692000000.0

    - Tangible Book Value: 236070000000.0

    - Invested Capital: 314041000000.0

    - Working Capital: 14315000000.0

    - Net Tangible Assets: 236070000000.0

    - Capital Lease Obligations: 79802000000.0

    - Common Stock Equity: 259151000000.0

    - Total Capitalization: 314041000000.0

    - Total Equity Gross Minority Interest: 259151000000.0

    - Stockholders Equity: 259151000000.0

    - Gains Losses Not Affecting Retained Earnings: -1918000000.0

    - Other Equity Adjustments: -1918000000.0

    - Treasury Stock: 7837000000.0

    - Retained Earnings: 152862000000.0

    - Additional Paid In Capital: 115934000000.0

    - Capital Stock: 110000000.0

    - Common Stock: 110000000.0

    - Preferred Stock: 0.0

    - Total Liabilities Net Minority Interest: 325475000000.0

    - Total Non Current Liabilities Net Minority Interest: 163998000000.0

    - Other Non Current Liabilities: 29306000000.0

    - Long Term Debt And Capital Lease Obligation: 134692000000.0

    - Long Term Capital Lease Obligation: 79802000000.0

    - Long Term Debt: 54890000000.0

    - Current Liabilities: 161477000000.0

    - Current Deferred Liabilities: 16305000000.0

    - Current Deferred Revenue: 16305000000.0

    - Payables And Accrued Expenses: 145172000000.0

    - Current Accrued Expenses: 60602000000.0

    - Payables: 84570000000.0

    - Accounts Payable: 84570000000.0

    - Total Non Current Assets: 408834000000.0

    - Other Non Current Assets: 71309000000.0

    - Goodwill And Other Intangible Assets: 23081000000.0

    - Other Intangible Assets: nan

    - Goodwill: 23081000000.0

    - Net PPE: 314444000000.0

    - Accumulated Depreciation: nan

    - Gross PPE: 314444000000.0

    - Construction In Progress: nan

    - Other Properties: 314444000000.0

    - Land And Improvements: nan

    - Properties: nan

    - Current Assets: 175792000000.0

    - Other Current Assets: nan

    - Inventory: 36103000000.0

    - Inventories Adjustments Allowances: -2700000000.0

    - Other Inventories: 38803000000.0

    - Finished Goods: nan

    - Receivables: 51638000000.0

    - Other Receivables: nan

    - Accounts Receivable: 51638000000.0

    - Allowance For Doubtful Accounts Receivable: -1900000000.0

    - Gross Accounts Receivable: 53538000000.0

    - Cash Cash Equivalents And Short Term Investments: 88051000000.0

    - Other Short Term Investments: 12960000000.0

    - Cash And Cash Equivalents: 75091000000.0

    '
  sentences:
  - Given Apple's negative working capital and substantial net debt, what specific
    factors within the current assets and current liabilities contribute most significantly
    to the company's short-term liquidity challenges as reflected in the balance sheet?
  - Given Amazon's balance sheet as of September 30, 2024, what percentage of total
    assets is represented by tangible assets (Net PPE + Goodwill and Other Intangible
    Assets + Inventory + Receivables + Cash Cash Equivalents And Short Term Investments)?
  - Given that Amazon's net debt as of June 30, 2023, was $13.563 billion and finished
    goods inventory was $36.587 billion, what could be inferred about Amazon's liquidity
    position and its ability to cover its net debt with its most readily saleable
    inventory?
- source_sentence: 'AAPL Balance Sheet as of 2024-12-31:

    - Total Assets: 344,085,000,000

    - Treasury Shares Number: nan

    - Ordinary Shares Number: 15037874000.0

    - Share Issued: 15037874000.0

    - Net Debt: 66500000000.0

    - Total Debt: 96799000000.0

    - Tangible Book Value: 66758000000.0

    - Invested Capital: 163557000000.0

    - Working Capital: -11125000000.0

    - Net Tangible Assets: 66758000000.0

    - Capital Lease Obligations: nan

    - Common Stock Equity: 66758000000.0

    - Total Capitalization: 150714000000.0

    - Total Equity Gross Minority Interest: 66758000000.0

    - Stockholders Equity: 66758000000.0

    - Gains Losses Not Affecting Retained Earnings: -6789000000.0

    - Other Equity Adjustments: -6789000000.0

    - Retained Earnings: -11221000000.0

    - Capital Stock: 84768000000.0

    - Common Stock: 84768000000.0

    - Total Liabilities Net Minority Interest: 277327000000.0

    - Total Non Current Liabilities Net Minority Interest: 132962000000.0

    - Other Non Current Liabilities: 49006000000.0

    - Tradeand Other Payables Non Current: nan

    - Long Term Debt And Capital Lease Obligation: 83956000000.0

    - Long Term Capital Lease Obligation: nan

    - Long Term Debt: 83956000000.0

    - Current Liabilities: 144365000000.0

    - Other Current Liabilities: 61151000000.0

    - Current Deferred Liabilities: 8461000000.0

    - Current Deferred Revenue: 8461000000.0

    - Current Debt And Capital Lease Obligation: 12843000000.0

    - Current Capital Lease Obligation: nan

    - Current Debt: 12843000000.0

    - Other Current Borrowings: 10848000000.0

    - Commercial Paper: 1995000000.0

    - Payables And Accrued Expenses: 61910000000.0

    - Payables: 61910000000.0

    - Total Tax Payable: nan

    - Income Tax Payable: nan

    - Accounts Payable: 61910000000.0

    - Total Non Current Assets: 210845000000.0

    - Other Non Current Assets: 77183000000.0

    - Non Current Deferred Assets: nan

    - Non Current Deferred Taxes Assets: nan

    - Investments And Advances: 87593000000.0

    - Investmentin Financial Assets: 87593000000.0

    - Available For Sale Securities: 87593000000.0

    - Net PPE: 46069000000.0

    - Accumulated Depreciation: -74546000000.0

    - Gross PPE: 120615000000.0

    - Leases: nan

    - Other Properties: nan

    - Machinery Furniture Equipment: nan

    - Land And Improvements: nan

    - Properties: nan

    - Current Assets: 133240000000.0

    - Other Current Assets: 13248000000.0

    - Inventory: 6911000000.0

    - Finished Goods: 4119000000.0

    - Raw Materials: 2792000000.0

    - Receivables: 59306000000.0

    - Other Receivables: 29667000000.0

    - Accounts Receivable: 29639000000.0

    - Cash Cash Equivalents And Short Term Investments: 53775000000.0

    - Other Short Term Investments: 23476000000.0

    - Cash And Cash Equivalents: 30299000000.0

    - Cash Equivalents: 3226000000.0

    - Cash Financial: 27073000000.0

    '
  sentences:
  - Given Microsoft's total assets and total liabilities as of 2024-12-31, what is
    the percentage of assets financed by liabilities?
  - Given Apple's significant cash and short-term investments, and negative working
    capital, how is the company managing its short-term liquidity and operational
    efficiency as of December 31, 2024?
  - What percentage of GOOGL's total assets as of March 31, 2024, are represented
    by tangible assets?
- source_sentence: 'META Balance Sheet as of 2024-03-31:

    - Total Assets: 222,844,000,000

    - Treasury Shares Number: nan

    - Ordinary Shares Number: 2537000000.0

    - Share Issued: 2537000000.0

    - Total Debt: 37633000000.0

    - Tangible Book Value: 128875000000.0

    - Invested Capital: 167916000000.0

    - Working Capital: 47229000000.0

    - Net Tangible Assets: 128875000000.0

    - Capital Lease Obligations: 19246000000.0

    - Common Stock Equity: 149529000000.0

    - Total Capitalization: 167916000000.0

    - Total Equity Gross Minority Interest: 149529000000.0

    - Stockholders Equity: 149529000000.0

    - Gains Losses Not Affecting Retained Earnings: -2655000000.0

    - Other Equity Adjustments: -2655000000.0

    - Retained Earnings: 76793000000.0

    - Additional Paid In Capital: 75391000000.0

    - Capital Stock: 0.0

    - Common Stock: 0.0

    - Total Liabilities Net Minority Interest: 73315000000.0

    - Total Non Current Liabilities Net Minority Interest: 45214000000.0

    - Other Non Current Liabilities: 1462000000.0

    - Tradeand Other Payables Non Current: 7795000000.0

    - Long Term Debt And Capital Lease Obligation: 35957000000.0

    - Long Term Capital Lease Obligation: 17570000000.0

    - Long Term Debt: 18387000000.0

    - Current Liabilities: 28101000000.0

    - Other Current Liabilities: 4909000000.0

    - Current Debt And Capital Lease Obligation: 1676000000.0

    - Current Capital Lease Obligation: 1676000000.0

    - Pensionand Other Post Retirement Benefit Plans Current: 3333000000.0

    - Payables And Accrued Expenses: 18183000000.0

    - Current Accrued Expenses: 9535000000.0

    - Payables: 8648000000.0

    - Dueto Related Parties Current: nan

    - Total Tax Payable: 4863000000.0

    - Accounts Payable: 3785000000.0

    - Total Non Current Assets: 147514000000.0

    - Other Non Current Assets: 8179000000.0

    - Investments And Advances: 6218000000.0

    - Investmentin Financial Assets: 6218000000.0

    - Available For Sale Securities: 6218000000.0

    - Goodwill And Other Intangible Assets: 20654000000.0

    - Other Intangible Assets: nan

    - Goodwill: 20654000000.0

    - Net PPE: 112463000000.0

    - Accumulated Depreciation: -35910000000.0

    - Gross PPE: 148373000000.0

    - Leases: 7079000000.0

    - Construction In Progress: 22975000000.0

    - Other Properties: 76922000000.0

    - Buildings And Improvements: 39322000000.0

    - Land And Improvements: 2075000000.0

    - Properties: 0.0

    - Current Assets: 75330000000.0

    - Other Current Assets: 3780000000.0

    - Receivables: 13430000000.0

    - Accounts Receivable: 13430000000.0

    - Cash Cash Equivalents And Short Term Investments: 58120000000.0

    - Other Short Term Investments: 25813000000.0

    - Cash And Cash Equivalents: 32307000000.0

    - Cash Equivalents: 25812000000.0

    - Cash Financial: 6495000000.0

    '
  sentences:
  - How does Microsoft's "Net Debt" compare to its "Stockholders Equity" as of March
    31, 2024, and what does this ratio suggest about the company's financial leverage?
  - How does META's reliance on capital leases, as indicated by the proportion of
    Long Term Capital Lease Obligations to Total Debt, impact the company's financial
    flexibility compared to relying solely on traditional long-term debt?
  - Given Amazon's Total Debt and Capital Lease Obligations, what proportion of their
    Long-Term Debt and Capital Lease Obligation is comprised of Capital Lease Obligations?
- source_sentence: 'META Balance Sheet as of 2024-09-30:

    - Total Assets: 256,408,000,000

    - Treasury Shares Number: nan

    - Ordinary Shares Number: 2524000000.0

    - Share Issued: 2524000000.0

    - Total Debt: 49047000000.0

    - Tangible Book Value: 143875000000.0

    - Invested Capital: 193352000000.0

    - Working Capital: 57737000000.0

    - Net Tangible Assets: 143875000000.0

    - Capital Lease Obligations: 20224000000.0

    - Common Stock Equity: 164529000000.0

    - Total Capitalization: 193352000000.0

    - Total Equity Gross Minority Interest: 164529000000.0

    - Stockholders Equity: 164529000000.0

    - Gains Losses Not Affecting Retained Earnings: -1192000000.0

    - Other Equity Adjustments: -1192000000.0

    - Retained Earnings: 84972000000.0

    - Additional Paid In Capital: 80749000000.0

    - Capital Stock: 0.0

    - Common Stock: 0.0

    - Total Liabilities Net Minority Interest: 91879000000.0

    - Total Non Current Liabilities Net Minority Interest: 58549000000.0

    - Other Non Current Liabilities: 2347000000.0

    - Tradeand Other Payables Non Current: 9171000000.0

    - Long Term Debt And Capital Lease Obligation: 47031000000.0

    - Long Term Capital Lease Obligation: 18208000000.0

    - Long Term Debt: 28823000000.0

    - Current Liabilities: 33330000000.0

    - Other Current Liabilities: 5705000000.0

    - Current Debt And Capital Lease Obligation: 2016000000.0

    - Current Capital Lease Obligation: 2016000000.0

    - Pensionand Other Post Retirement Benefit Plans Current: 5458000000.0

    - Payables And Accrued Expenses: 20151000000.0

    - Current Accrued Expenses: 9792000000.0

    - Payables: 10359000000.0

    - Dueto Related Parties Current: nan

    - Total Tax Payable: 2703000000.0

    - Accounts Payable: 7656000000.0

    - Total Non Current Assets: 165341000000.0

    - Other Non Current Assets: 11642000000.0

    - Investments And Advances: 6071000000.0

    - Investmentin Financial Assets: 6071000000.0

    - Available For Sale Securities: 6071000000.0

    - Goodwill And Other Intangible Assets: 20654000000.0

    - Other Intangible Assets: nan

    - Goodwill: 20654000000.0

    - Net PPE: 126974000000.0

    - Accumulated Depreciation: -40854000000.0

    - Gross PPE: 167828000000.0

    - Leases: 7246000000.0

    - Construction In Progress: 23301000000.0

    - Other Properties: 89775000000.0

    - Buildings And Improvements: 45392000000.0

    - Land And Improvements: 2114000000.0

    - Properties: 0.0

    - Current Assets: 91067000000.0

    - Other Current Assets: 5467000000.0

    - Receivables: 14700000000.0

    - Accounts Receivable: 14700000000.0

    - Cash Cash Equivalents And Short Term Investments: 70900000000.0

    - Other Short Term Investments: 27048000000.0

    - Cash And Cash Equivalents: 43852000000.0

    - Cash Equivalents: 37323000000.0

    - Cash Financial: 6529000000.0

    '
  sentences:
  - What percentage of Microsoft's total assets as of December 31, 2023, are comprised
    of goodwill and other intangible assets?
  - Given Amazon's total capitalization, long-term debt, and capital lease obligations
    as of December 31, 2023, what is the proportion of total capitalization financed
    by debt (both long-term and capital leases)?
  - What is the ratio of Meta's total debt to its stockholders' equity as of September
    30, 2024?
- source_sentence: 'AAPL Balance Sheet as of 2023-12-31:

    - Total Assets: 353,514,000,000

    - Treasury Shares Number: 0.0

    - Ordinary Shares Number: 15460223000.0

    - Share Issued: 15460223000.0

    - Net Debt: 67280000000.0

    - Total Debt: 108040000000.0

    - Tangible Book Value: 74100000000.0

    - Invested Capital: 182140000000.0

    - Working Capital: 9719000000.0

    - Net Tangible Assets: 74100000000.0

    - Capital Lease Obligations: nan

    - Common Stock Equity: 74100000000.0

    - Total Capitalization: 169188000000.0

    - Total Equity Gross Minority Interest: 74100000000.0

    - Stockholders Equity: 74100000000.0

    - Gains Losses Not Affecting Retained Earnings: -9378000000.0

    - Other Equity Adjustments: -9378000000.0

    - Retained Earnings: 8242000000.0

    - Capital Stock: 75236000000.0

    - Common Stock: 75236000000.0

    - Total Liabilities Net Minority Interest: 279414000000.0

    - Total Non Current Liabilities Net Minority Interest: 145441000000.0

    - Other Non Current Liabilities: 50353000000.0

    - Tradeand Other Payables Non Current: nan

    - Long Term Debt And Capital Lease Obligation: 95088000000.0

    - Long Term Capital Lease Obligation: nan

    - Long Term Debt: 95088000000.0

    - Current Liabilities: 133973000000.0

    - Other Current Liabilities: 54611000000.0

    - Current Deferred Liabilities: 8264000000.0

    - Current Deferred Revenue: 8264000000.0

    - Current Debt And Capital Lease Obligation: 12952000000.0

    - Current Capital Lease Obligation: nan

    - Current Debt: 12952000000.0

    - Other Current Borrowings: 10954000000.0

    - Commercial Paper: 1998000000.0

    - Payables And Accrued Expenses: 58146000000.0

    - Payables: 58146000000.0

    - Total Tax Payable: nan

    - Income Tax Payable: nan

    - Accounts Payable: 58146000000.0

    - Total Non Current Assets: 209822000000.0

    - Other Non Current Assets: 66681000000.0

    - Non Current Deferred Assets: nan

    - Non Current Deferred Taxes Assets: nan

    - Investments And Advances: 99475000000.0

    - Investmentin Financial Assets: 99475000000.0

    - Available For Sale Securities: 99475000000.0

    - Net PPE: 43666000000.0

    - Accumulated Depreciation: -72510000000.0

    - Gross PPE: 116176000000.0

    - Leases: nan

    - Other Properties: nan

    - Machinery Furniture Equipment: nan

    - Land And Improvements: nan

    - Properties: nan

    - Current Assets: 143692000000.0

    - Other Current Assets: 13979000000.0

    - Inventory: 6511000000.0

    - Finished Goods: nan

    - Raw Materials: nan

    - Receivables: 50102000000.0

    - Other Receivables: 26908000000.0

    - Accounts Receivable: 23194000000.0

    - Cash Cash Equivalents And Short Term Investments: 73100000000.0

    - Other Short Term Investments: 32340000000.0

    - Cash And Cash Equivalents: 40760000000.0

    - Cash Equivalents: 11218000000.0

    - Cash Financial: 29542000000.0

    '
  sentences:
  - Given Apple's negative working capital and substantial net debt, what percentage
    of total assets are financed by debt (Total Debt / Total Assets)?
  - Given the balance sheet, what percentage of Microsoft's total assets as of June
    30, 2024 are represented by goodwill and other intangible assets?
  - Given Apple's substantial cash and short-term investments, and a significant level
    of debt, what is the rationale behind maintaining such a large net debt position
    ($67.28 billion) instead of utilizing available liquid assets to reduce it?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'AAPL Balance Sheet as of 2023-12-31:\n- Total Assets: 353,514,000,000\n- Treasury Shares Number: 0.0\n- Ordinary Shares Number: 15460223000.0\n- Share Issued: 15460223000.0\n- Net Debt: 67280000000.0\n- Total Debt: 108040000000.0\n- Tangible Book Value: 74100000000.0\n- Invested Capital: 182140000000.0\n- Working Capital: 9719000000.0\n- Net Tangible Assets: 74100000000.0\n- Capital Lease Obligations: nan\n- Common Stock Equity: 74100000000.0\n- Total Capitalization: 169188000000.0\n- Total Equity Gross Minority Interest: 74100000000.0\n- Stockholders Equity: 74100000000.0\n- Gains Losses Not Affecting Retained Earnings: -9378000000.0\n- Other Equity Adjustments: -9378000000.0\n- Retained Earnings: 8242000000.0\n- Capital Stock: 75236000000.0\n- Common Stock: 75236000000.0\n- Total Liabilities Net Minority Interest: 279414000000.0\n- Total Non Current Liabilities Net Minority Interest: 145441000000.0\n- Other Non Current Liabilities: 50353000000.0\n- Tradeand Other Payables Non Current: nan\n- Long Term Debt And Capital Lease Obligation: 95088000000.0\n- Long Term Capital Lease Obligation: nan\n- Long Term Debt: 95088000000.0\n- Current Liabilities: 133973000000.0\n- Other Current Liabilities: 54611000000.0\n- Current Deferred Liabilities: 8264000000.0\n- Current Deferred Revenue: 8264000000.0\n- Current Debt And Capital Lease Obligation: 12952000000.0\n- Current Capital Lease Obligation: nan\n- Current Debt: 12952000000.0\n- Other Current Borrowings: 10954000000.0\n- Commercial Paper: 1998000000.0\n- Payables And Accrued Expenses: 58146000000.0\n- Payables: 58146000000.0\n- Total Tax Payable: nan\n- Income Tax Payable: nan\n- Accounts Payable: 58146000000.0\n- Total Non Current Assets: 209822000000.0\n- Other Non Current Assets: 66681000000.0\n- Non Current Deferred Assets: nan\n- Non Current Deferred Taxes Assets: nan\n- Investments And Advances: 99475000000.0\n- Investmentin Financial Assets: 99475000000.0\n- Available For Sale Securities: 99475000000.0\n- Net PPE: 43666000000.0\n- Accumulated Depreciation: -72510000000.0\n- Gross PPE: 116176000000.0\n- Leases: nan\n- Other Properties: nan\n- Machinery Furniture Equipment: nan\n- Land And Improvements: nan\n- Properties: nan\n- Current Assets: 143692000000.0\n- Other Current Assets: 13979000000.0\n- Inventory: 6511000000.0\n- Finished Goods: nan\n- Raw Materials: nan\n- Receivables: 50102000000.0\n- Other Receivables: 26908000000.0\n- Accounts Receivable: 23194000000.0\n- Cash Cash Equivalents And Short Term Investments: 73100000000.0\n- Other Short Term Investments: 32340000000.0\n- Cash And Cash Equivalents: 40760000000.0\n- Cash Equivalents: 11218000000.0\n- Cash Financial: 29542000000.0\n',
    "Given Apple's substantial cash and short-term investments, and a significant level of debt, what is the rationale behind maintaining such a large net debt position ($67.28 billion) instead of utilizing available liquid assets to reduce it?",
    "Given the balance sheet, what percentage of Microsoft's total assets as of June 30, 2024 are represented by goodwill and other intangible assets?",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 28 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 28 samples:
  |         | sentence_0                                                                           | sentence_1                                                                        |
  |:--------|:-------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                               | string                                                                            |
  | details | <ul><li>min: 256 tokens</li><li>mean: 256.0 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 42.5 tokens</li><li>max: 66 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | sentence_1                                                                                                                                                                                                                                                              |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>AAPL Balance Sheet as of 2024-12-31:<br>- Total Assets: 344,085,000,000<br>- Treasury Shares Number: nan<br>- Ordinary Shares Number: 15037874000.0<br>- Share Issued: 15037874000.0<br>- Net Debt: 66500000000.0<br>- Total Debt: 96799000000.0<br>- Tangible Book Value: 66758000000.0<br>- Invested Capital: 163557000000.0<br>- Working Capital: -11125000000.0<br>- Net Tangible Assets: 66758000000.0<br>- Capital Lease Obligations: nan<br>- Common Stock Equity: 66758000000.0<br>- Total Capitalization: 150714000000.0<br>- Total Equity Gross Minority Interest: 66758000000.0<br>- Stockholders Equity: 66758000000.0<br>- Gains Losses Not Affecting Retained Earnings: -6789000000.0<br>- Other Equity Adjustments: -6789000000.0<br>- Retained Earnings: -11221000000.0<br>- Capital Stock: 84768000000.0<br>- Common Stock: 84768000000.0<br>- Total Liabilities Net Minority Interest: 277327000000.0<br>- Total Non Current Liabilities Net Minority Interest: 132962000000.0<br>- Other Non Current Liabilities: 49006000000.0<br>- Tradeand Other Payables Non Current: nan<br>- Long Term Deb...</code> | <code>Given Apple's significant cash and short-term investments, and negative working capital, how is the company managing its short-term liquidity and operational efficiency as of December 31, 2024?</code>                                                          |
  | <code>AAPL Balance Sheet as of 2024-09-30:<br>- Total Assets: 364,980,000,000<br>- Treasury Shares Number: nan<br>- Ordinary Shares Number: 15116786000.0<br>- Share Issued: 15116786000.0<br>- Net Debt: 76686000000.0<br>- Total Debt: 106629000000.0<br>- Tangible Book Value: 56950000000.0<br>- Invested Capital: 163579000000.0<br>- Working Capital: -23405000000.0<br>- Net Tangible Assets: 56950000000.0<br>- Capital Lease Obligations: nan<br>- Common Stock Equity: 56950000000.0<br>- Total Capitalization: 142700000000.0<br>- Total Equity Gross Minority Interest: 56950000000.0<br>- Stockholders Equity: 56950000000.0<br>- Gains Losses Not Affecting Retained Earnings: -7172000000.0<br>- Other Equity Adjustments: -7172000000.0<br>- Retained Earnings: -19154000000.0<br>- Capital Stock: 83276000000.0<br>- Common Stock: 83276000000.0<br>- Total Liabilities Net Minority Interest: 308030000000.0<br>- Total Non Current Liabilities Net Minority Interest: 131638000000.0<br>- Other Non Current Liabilities: 36634000000.0<br>- Tradeand Other Payables Non Current: 9254000000.0<br>- Lon...</code> | <code>Given Apple's negative working capital and substantial net debt, what percentage of total assets are financed by debt (Total Debt / Total Assets)?</code>                                                                                                         |
  | <code>AAPL Balance Sheet as of 2024-06-30:<br>- Total Assets: 331,612,000,000<br>- Treasury Shares Number: nan<br>- Ordinary Shares Number: 15222259000.0<br>- Share Issued: 15222259000.0<br>- Net Debt: 75739000000.0<br>- Total Debt: 101304000000.0<br>- Tangible Book Value: 66708000000.0<br>- Invested Capital: 168012000000.0<br>- Working Capital: -6189000000.0<br>- Net Tangible Assets: 66708000000.0<br>- Capital Lease Obligations: nan<br>- Common Stock Equity: 66708000000.0<br>- Total Capitalization: 152904000000.0<br>- Total Equity Gross Minority Interest: 66708000000.0<br>- Stockholders Equity: 66708000000.0<br>- Gains Losses Not Affecting Retained Earnings: -8416000000.0<br>- Other Equity Adjustments: -8416000000.0<br>- Retained Earnings: -4726000000.0<br>- Capital Stock: 79850000000.0<br>- Common Stock: 79850000000.0<br>- Total Liabilities Net Minority Interest: 264904000000.0<br>- Total Non Current Liabilities Net Minority Interest: 133280000000.0<br>- Other Non Current Liabilities: 47084000000.0<br>- Tradeand Other Payables Non Current: nan<br>- Long Term Debt...</code> | <code>Given Apple's negative working capital and substantial net debt, what specific factors within the current assets and current liabilities contribute most significantly to the company's short-term liquidity challenges as reflected in the balance sheet?</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1000
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1000
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 250.0  | 500  | 1.2725        |
| 500.0  | 1000 | 0.3229        |
| 750.0  | 1500 | 0.0184        |
| 1000.0 | 2000 | 0.0026        |


### Framework Versions
- Python: 3.9.13
- Sentence Transformers: 3.4.1
- Transformers: 4.48.3
- PyTorch: 2.2.2
- Accelerate: 1.3.0
- Datasets: 3.3.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->