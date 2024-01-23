FASTgres consists of the following main functionalities:

- **`Context classification`**: Queries are parsed and then categorized into a context (i.e., bucket). 
Within each context, a gradient boosting model is trained and used to predict the best fitting query 
optimizer instruction for each query.
- **`Query encoding`**: To obtain a learnable representation, we deploy an encoding method (i.e., featurization) 
within each context. Such encoding method allows to tranform SQL queries into numerical vector representations. 
To encode a query, database statistics like filter cardinalities are used.
- **`Labeling`**: This method is used to obtain the best query optimizer hints that can be learned by our 
gradient boosting classifier. Labels are represented as an integer value depicting their binary encoding. 
For FASTgres' current layout, we deploy `six` hints, namely **2<sup>6</sup>** hint set combinations. 
These include the main 3 operators for scans and joins used in PostgreSQL [^1].
In such encoding `0` stands for `disabled` hint and `1` for `enabled`. E.g., an integer representation `63`
depicts the binary value of `1 1 1 1 1 1` meaning every optimizer hint is enabled. We also refer to this configuration
 as the PostgreSQL default hint set as this is the configuration that is used by PostgreSQL if no 
manual hinting is performed.
- **`Training`**: Once queries have been labeled, they can be used for training a gradient boosting model. 
Again, within each context, one model is trained using encoded queries and their respective labels for supervised
 learning.
- **`Inference`**: After training, context models are loaded and used for prediction on unseen queries.
- **`Retraining`**: One aspect FASTgres employs is the possibility to utilize retraining based on observed experiences. 
Here, a timeout threshold is chosen. Once a query surpasses such timeout threshold, it is used for labeling and 
retraining the context model which it belongs to.