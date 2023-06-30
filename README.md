# SeeFood101

## Part 1

My evolving first personal in deep learning, specifically in image classification. The biggest objective is to teach me PyTorch. This can be divided into finer objectives:

1. Finish the process of training a model end-to-end, which includes:
   1. Get the data.
   2. Get the model.
   3. Run the train + validation loop.
   4. Save the model.
   5. All within Jupyter Notebook.
2. Fine-tune a pre-trained model instead of training from scratch.
3. Export the useful helper functions into a folder for reusability.
4. Use the most basic tool i.e., TensorBoard to track the training result.
5. Replicate (the architecture, not the training data or computing resource) one research paper.
6. Deploy the model to the simplest option: HuggingFace Space (and keep it alive through a Discord Bot).

**Update 30/06/2023**: Part 1 is completed. I am officially confident about implementing PyTorch, in the sense that now I know enough to get started on any project and trust that I can find my way through if I am stuck. 

The model is deployed at https://huggingface.co/spaces/HangenYuu/SeeFood101v1.

Let's start on Part 2.

## Part 2

> Machine learning engineering is 10% machine learning and 90% engineering.
>
> *[Chip Huyen](https://twitter.com/chipro/status/1315678863347920896?lang=en), my intellectual hero*

In production i.e., where it actually matters for many people, training and saving a model on its own does not matter. The details can be summarized into this table taken from [Designing Machine Learning Systems](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch01.html#key_differences_between_ml_in_research) from Huyen Chip.

|   | Research | Production |
| --- | --- | --- |
| **Requirements** | State-of-the-art model performance on benchmark datasets | Different stakeholders have different requirements |
| **Computational priority** | Fast training, high throughput | Fast inference, low latency |
| **Data** | Static | Constantly shifting |
| **Fairness** | Often not a focus | Must be considered |
| **Interpretability** | Often not a focus | Must be considered |

Hence, for someone wanting to become an MLE (such as the writer), learning how to train a model is not enough. I also need to learn how to **deploy**, **monitor**, and **maintain** the model.

