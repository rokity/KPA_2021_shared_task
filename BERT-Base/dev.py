from transformers import DataCollatorWithPadding,TrainingArguments,default_data_collator,EvalPrediction,AutoModelForSequenceClassification,AutoTokenizer,Trainer
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate
from datasets import Dataset as ds



access_token = "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"
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def load_kpm_data(gold_data_dir, subset, submitted_kp_file=None):
    print("\nֿ** loading task data:")
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    if not submitted_kp_file:
        key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    else:
        key_points_file=submitted_kp_file
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")


    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)

    return arguments_df, key_points_df, labels_file_df

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
accuracy = evaluate.load("accuracy")

def preprocess_function(examples):
    # print(examples.info())
    token = tokenizer(examples[4]+examples[5],examples[3],
     truncation=True)
    token["labels"] = examples[2]
    # print(token['labels'])
    # print(examples.keys())
    return token

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    
    # config = BertConfig.from_pretrained("bert-base-cased", num_labels=1)
    # ,hidden_act='silu')
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",num_labels=2, id2label=id2label, label2id=label2id)
    gold_data_dir = 'kpm_data/'
    # Merge Labels with arguments and key_points for training dataset
    arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset="train")
    labels_df_dev=labels_df.merge(kp_df,on="key_point_id",how="left")
    labels_df_dev=labels_df_dev.merge(arg_df,on='arg_id',how="left")
    labels_df_dev.drop(["stance_x", "topic_x"], axis=1, inplace=True)
    labels_df_dev=labels_df_dev.rename(columns={"stance_y":"stance", "topic_y":"topic"})
    # Merge Labels with arguments and key_points for test dataset
    arg_df_test, kp_df_test, labels_df_test = load_kpm_data(gold_data_dir, subset="dev")
    labels_df_test=labels_df_test.merge(kp_df_test,on="key_point_id",how="left")
    labels_df_test=labels_df_test.merge(arg_df_test,on='arg_id',how="left")
    labels_df_test.drop(["stance_x", "topic_x"], axis=1, inplace=True)
    labels_df_test=labels_df_test.rename(columns={"stance_y":"stance", "topic_y":"topic"})
    #encode each sentence and append to dictionary
    # tokens_dev = tokenizer.batch_encode_plus(labels_df_dev["argument"].to_list(), padding="max_length", truncation=True)
    # labels_df_dev= ds.from_pandas(labels_df_dev)
    # print(labels_df_dev.columns)
    result_train = labels_df_dev.apply(preprocess_function,axis=1)
    # result_train=pd.DataFrame(result_train.to_list())
    # input_ids = torch.tensor(result_train['input_ids']).squeeze()  
    # input_mask = torch.tensor(result_train['attention_mask']).squeeze()
    # input_stance = torch.tensor(labels_df_dev["stance"]).squeeze()
    # input_label = torch.tensor(labels_df_dev["label"]).squeeze()
    # tokenized_train_dataset = TensorDataset(input_ids, input_mask, input_stance, input_label)
    # tokenized_train_dataset = DataLoader(tokenized_train_dataset, batch_size=16, shuffle=True)

    result_test = labels_df_test.apply(preprocess_function,axis=1)
    # result_test=pd.DataFrame(result_test.to_list())
    # input_ids = torch.tensor(result_test['input_ids']).squeeze()  
    # input_mask = torch.tensor(result_test['attention_mask']).squeeze()
    # input_stance = torch.tensor(labels_df_test["stance"]).squeeze()
    # input_label = torch.tensor(labels_df_test["label"]).squeeze()
    # tokenized_dev_dataset = TensorDataset(input_ids, input_mask, input_stance, input_label)
    # tokenized_dev_dataset = DataLoader(tokenized_dev_dataset, batch_size=16, shuffle=True)
    # # result_train["label"] = labels_df_dev["label"]
    # train_dataset = result_train
    # labels_df_test= ds.from_pandas(labels_df_test)
    # # result_test = tokenizer(labels_df_test["argument"],  truncation=True)
    # result_test = labels_df_test.apply(preprocess_function, axis=1)
    # # result_test["label"] = labels_df_test["label"]
    # eval_dataset = result_test
    training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=result_train,
    eval_dataset=result_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
    # print(type(tokenized_train_dataset))
    train_result = trainer.train()
    metrics = train_result.metrics
    max_train_samples = (len(result_train))
    metrics["train_samples"] = min(max_train_samples, len(result_train))
    trainer.log_metrics("train", metrics)
    

    # train_dataset = Dataset(tokens_dev, labels_df_dev["label"])
#     train_dataset = tf.data.Dataset.from_tensor_slices((
#       dict(tokens_dev),
#       np.asarray(labels_df_dev["label"]).astype('int32').reshape((len(labels_df_dev["label"]),1))
#   ))
    
    # tokens_test = tokenizer.batch_encode_plus(labels_df_test["argument"].to_list(), padding="max_length", truncation=True)
    # test_dataset = Dataset(tokens_test, labels_df_test["label"])
#     test_dataset = tf.data.Dataset.from_tensor_slices((
#       dict(tokens_test),
#       np.asarray(labels_df_test["label"]).astype('int32').reshape((len(labels_df_test["label"]),1))
#   ))
    

#     model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
#     loss='binary_crossentropy',
#     metrics=["acc"],
# )
#     # model.add(tf.layer.Dense(512, activation='sigmoid'))
#     print(model.summary())
#     model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16,validation_data=test_dataset.shuffle(1000).batch(16))
#     model.save_pretrained("bert-base-cased")
#     tokenizer.save_pretrained("bert-base-cased")

    
    
   