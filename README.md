# paper-classfication
# 樣型識別期末報告

# 論文分類

# B0529001 馮少迪



Data是從參加這個比賽下載

 

最後訓練n個epoch的結果，資料是分成5600/1400，因為不知道怎麼微調，感覺已經到極限了。不過這個結果用來跑test data 是過擬合的。

Code 鏈接：

[https://colab.research.google.com/drive/1ZzZuKK3UbGqNQweqT90lKBNSJ5C29zcX](https://colab.research.google.com/drive/1ZzZuKK3UbGqNQweqT90lKBNSJ5C29zcX)



In this notebook, we&#39;ll be exploring how to use BERT with fastai for sentence classification.

_import_ _numpy_ _as_ _np_
_import_ _pandas_ _as_ _pd_

_from_ _pathlib_ _import_ _Path_
_from_ _typing_ _import_ _\*_

_import_ _torch_
_import_ _torch.optim_ _as_ _optim_

_from_ _fastai_ _import_ _\*_
_from_ _fastai.vision_ _import_ _\*_
_from_ _fastai.text_ _import_ _\*_
_from_ _fastai.callbacks_ _import_ _\*_

**class** _Config( __dict__ ):_
 _   _ **def** _\_\_init\_\___(__self__, \*\*kwargs):_
 _        __super__ (). __\_\_init\_\___ (\*\*kwargs)_
 _       _ **for** _k, v in kwargs.items():_
 _            __setattr__ ( __self__ , k, v)_
 _   _
 _   _ **def** _set__(__self__, key, val):_
 _        __self__ [key] = val_
 _        __setattr__ ( __self__ , key, val)_

_config = Config(_
 _   testing= __False__ ,_
 _   bert\_model\_name= __&quot;bert-base-uncased&quot;__ ,_
 _   max\_lr= __3e-5__ ,_
 _   epochs= __1__ ,_
 _   use\_fp16= __False__ ,_
 _   bs= __4__ ,_
 _   discriminative= __False__ ,_
 _   max\_seq\_len= __128__ ,_
_)_

We&#39;ll be using the pytorch-pretrained-bert package, so install it if you do not have it yet!

BERT requires a special wordpiece tokenizer and a vocabulary to go along with that. Thankfully, the pytorch-pretrained-bert package provides all of that within the handy _BertTokenizer_ class.

_!pip install pytorch\_pretrained\_bert_

# Data

BERT使用自己的词件标记器。 BERT需要将[CLS]和[SEP]令牌添加到每个序列。 BERT使用自己的预建词汇表。 让我们看一下如何处理这些问题。

使用字词标记器并处理特殊标记

编写我们自己的词件标记器并处理从词件到id的映射将是一个很大的麻烦。幸运的是，出色的pytorch-pretrained-bert软件包在其BertTokenizer中为我们提供了所有必要的信息。

_from_ _pytorch\_pretrained\_bert_ _import_ _BertTokenizer_
_bert\_tok = BertTokenizer.from\_pretrained(_
 _   config.bert\_model\_name,_
_)_

_100%|██████████| 231508/231508 [00:00\&lt;00:00, 430829.40B/s]_

FastAI has its own conventions for handling tokenization, so we&#39;ll need to wrap the tokenizer within a different class. This is a bit confusing but shouldn&#39;t be that much of a hassle.

Notice we add the [CLS] and [SEP] special tokens to the start and end of the sequence here.

BERT具有多种风格，因此我们为类传递了将要使用的BERT模型的名称（在本文中，我们将使用无大小写的较小版本）。

Fastai有关于标记化的内部约定，因此我们将此标记化器包装在其自己的Tokenizer类中。这有点令人困惑，但不应该太麻烦。 Fastai有关于标记化的内部约定，因此我们将此标记化器包装在其自己的Tokenizer类中。这有点令人困惑，但不应该太麻烦。

**class** _FastAiBertTokenizer(BaseTokenizer):_
 _   _&quot;&quot;&quot;Wrapper around BertTokenizer to be compatible with fast.ai&quot;&quot;&quot;
 _   _ **def** _\_\_init\_\___(__self __, tokenizer: BertTokenizer, max\_seq\_len:_ _int__ = __128__ , \*\*kwargs):_
 _        __self__.\_pretrained\_tokenizer = tokenizer_
 _        __self__.max\_seq\_len = max\_seq\_len_

 _   _ **def** _\_\_call\_\___(__self__, \*args, \*\*kwargs):_
 _       _ **return** _self_

 _   _ **def** _tokenizer( __self__ , t: __str__ ) -\&gt; List[__str__]:_
 _       _&quot;&quot;&quot;Limits the maximum sequence length&quot;&quot;&quot;
 _       _ **return** _[__&quot;[CLS]&quot;__] +_ _self__.\_pretrained\_tokenizer.tokenize(t)[:__self __.max\_seq\_len -_ _2__] + [__&quot;[SEP]&quot;__]_

Slightly confusingly, we further need to wrap the tokenizer above in a _Tokenizer_ object. 如您所见，我们将在此处添加[CLS]和[SEP]令牌，并限制令牌化序列的长度。

有点令人困惑，我们需要将以上代码包装在另一个Tokenizer中，以传递给预处理器。

_fastai\_tokenizer = Tokenizer(tok\_func=FastAiBertTokenizer(bert\_tok, max\_seq\_len=config.max\_seq\_len), pre\_rules=[], post\_rules=[])_

Now, we need to make sure fastai uses the same mapping from wordpiece to integer as BERT originally did. Again, fastai has its own conventions on vocabulary so we&#39;ll be passing the vocabulary internal to the _BertTokenizer_ and constructing a fastai _Vocab_ object to use for preprocessing.

使用BERT词汇

bert令牌生成器还包含词汇，作为从单词到id的字典映射。与令牌生成器一样，由于fastai具有自己的词汇约定，因此我们需要根据bert词汇构造一个fastai Vocab对象。幸运的是，这很简单–我们可以简单地通过在词汇表中传递标记列表来做到这一点。

_fastai\_bert\_vocab = Vocab( __list__ (bert\_tok.vocab.keys()))_

Now we have all the pieces we need to make BERT work with fastai! We&#39;ll load the data into dataframes and construct a validation set.

放在一起

现在，我们拥有构建数据束所需的一切。在本教程中，我们将从数据帧构建数据绑定，但是有多种加载数据的方法（请参阅官方文档）。

首先，我们使用熊猫将数据加载到数据帧中，并在其中处理验证集。

_from_ _sklearn.model\_selection_ _import_ _train\_test\_split_

_DATA\_ROOT = Path( __&quot;&quot;__ ) /_ _&quot;data&quot;_ _/_ _&quot;jigsaw&quot;_

_train, test = [pd.read\_csv(DATA\_ROOT / fname)_ **for** _fname in [__&quot;traindata.csv&quot;__ ,_ _&quot;testdata.csv&quot;__]]_
_train, val = train\_test\_split(train)_

**if** _config.testing:_
 _   train = train.head( __1024__ )_
 _   val = val.head( __1024__ )_
 _   test = test.head( __1024__ )_

Now, we can build the databunch using the tokenizer and vocabulary we build above. Notice we&#39;re passing the _include\_bos=False_ and _include\_eos=False_ options. This is to prevent fastai from adding its own SOS/EOS tokens that will interfere with BERT&#39;s SOS/EOS tokens.

现在，我们在TextDataBunch上调用from\_df方法 注意，我们传递了include\_bos = False和include\_eos = False选项。这是因为fastai默认情况下会添加自己的bos和eos令牌，这会干扰BERT添加的[CLS]和[SEP]令牌。请注意，此选项是新选项，可能不适用于旧版本的fastai。

_label\_cols = [__&quot;THEORETICAL&quot;__ ,_ _&quot;ENGINEERING&quot; __,_ _&quot;EMPIRICAL&quot;__ ,_ _&quot;OTHERS&quot;__]_

_databunch = TextDataBunch.from\_df( __&quot;.&quot;__ , train, val, test,_
 _                 tokenizer=fastai\_tokenizer,_
 _                 vocab=fastai\_bert\_vocab,_
 _                 include\_bos= __False__ ,_
 _                 include\_eos= __False__ ,_
 _                 text\_cols= __&quot;Title&quot;__ ,_
 _                 label\_cols=label\_cols,_
 _                 bs=config.bs,_
 _                 collate\_fn=partial(pad\_collate, pad\_first= __False__ , pad\_idx= __0__ ),_
 _            )_

Alternatively, we can pass our own list of Preprocessors to the databunch (this is effectively what is happening behind the scenes)

让我们更深入地了解幕后到底发生了什么。上面的代码使用词片标记器和BERT词汇表初始化TokenizerProcessor和NumericizeProcessor，然后将其应用于数据框中的每个文本。

实际上，我们还可以初始化自己的TokenizerProcessor和NumericizeProcessor并将它们传递给数据束，而不是传递配置。

**class** _BertTokenizeProcessor(TokenizeProcessor):_
 _   _ **def** _\_\_init\_\___(__self__, tokenizer):_
 _        __super__ (). __\_\_init\_\___ (tokenizer=tokenizer, include\_bos= __False__ , include\_eos= __False__ )_

**class** _BertNumericalizeProcessor(NumericalizeProcessor):_
 _   _ **def** _\_\_init\_\___(__self__, \*args, \*\*kwargs):_
 _        __super__ (). __\_\_init\_\___ (\*args, vocab=Vocab( __list__ (bert\_tok.vocab.keys())), \*\*kwargs)_

**def** _get\_bert\_processor(tokenizer:Tokenizer= __None__ , vocab:Vocab= __None__ ):_
 _   _&quot;&quot;&quot;
    Constructing preprocessors for BERT
    We remove sos/eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original BERT model.
    &quot;&quot;&quot;
 _   _ **return** _[BertTokenizeProcessor(tokenizer=tokenizer),_
 _           NumericalizeProcessor(vocab=vocab)]_

To use our own custom preprocessors, we&#39;ll need to modify the _from\_df_ method to call the function above. Not the cleanest code but it will suffice.

我们只是包装标记器和词汇表，然后将它们放到管道中以预处理文本。要使用此管道，我们需要稍微修改数据绑定代码以在内部调用get\_bert\_processor。

**class** _BertDataBunch(TextDataBunch):_
 _   __@classmethod_
 _   _ **def** _from\_df(cls, path:PathOrStr, train\_df:DataFrame, valid\_df:DataFrame, test\_df:Optional[DataFrame]= __None__ ,_
 _               tokenizer:Tokenizer= __None__ , vocab:Vocab= __None__ , classes:Collection[__str__]= __None__ , text\_cols:IntsOrStrs= __1__ ,_
 _               label\_cols:IntsOrStrs= __0__ , label\_delim: __str__ = __None__ , \*\*kwargs) -\&gt; DataBunch:_
 _       _&quot;Create a `TextDataBunch` from DataFrames.&quot;
 _       p\_kwargs, kwargs = split\_kwargs\_by\_func(kwargs, get\_bert\_processor)_
 _       _# use our custom processors while taking tokenizer and vocab as kwargs
 _       processor = get\_bert\_processor(tokenizer=tokenizer, vocab=vocab, \*\*p\_kwargs)_
 _       _ **if** _classes is_ _None_ _and is\_listy(label\_cols) and_ _len__(label\_cols) \&gt;_ _1__: classes = label\_cols_
 _       src = ItemLists(path, TextList.from\_df(train\_df, path, cols=text\_cols, processor=processor),_
 _                       TextList.from\_df(valid\_df, path, cols=text\_cols, processor=processor))_
 _       src = src.label\_for\_lm()_ **if** _cls==TextLMDataBunch_ **else** _src.label\_from\_df(cols=label\_cols, classes=classes)_
 _       _ **if** _test\_df is not_ _None__: src.add\_test(TextList.from\_df(test\_df, path, cols=text\_cols))_
 _       _ **return** _src.databunch(\*\*kwargs)_

# 现在，我们可以像这样构建数据仓库。
# this will produce a virtually identical databunch to the code above
# databunch = BertDataBunch.from\_df(&quot;.&quot;, train, val, test,
#                   tokenizer=fastai\_tokenizer,
#                   vocab=fastai\_bert\_vocab,
#                   text\_cols=&quot;comment\_text&quot;,
#                   label\_cols=label\_cols,
#                   bs=config.bs,
#                   collate\_fn=partial(pad\_collate, pad\_first=False, pad\_idx=0),
#              )

# Model

Now with the data in place, we will prepare the model and loss functions. Again, the pytorch-pretrained-bert package gives us a sequence classifier based on BERT straight out of the box.

_from_ _pytorch\_pretrained\_bert.modeling_ _import_ _BertConfig, BertForSequenceClassification_
_bert\_model = BertForSequenceClassification.from\_pretrained(config.bert\_model\_name, num\_labels= __4__ )_

Since this is a multilabel classification problem, we&#39;re using _BCEWithLogitsLoss_

_loss\_func = nn.BCEWithLogitsLoss()_
#about bcewithlogistloss https://blog.csdn.net/qq\_22210253/article/details/85222093

And now we can build the _Learner_.

_from_ _fastai.callbacks_ _import_ _\*_

_learner = Learner(_
 _   databunch, bert\_model,_
 _   loss\_func=loss\_func,_
_)_

And we&#39;re done! All the rest is fastai magic. For example, you can use half-precision training simply by calling _learner.to\_fp16()_

#我们还可以利用fastai必须提供的其他一些功能。例如，我们可以通过以下代码轻松地使用半精度训练。
**if** _config.use\_fp16: learner = learner.to\_fp16()_

We can also use the learning rate finder.

_learner.lr\_find()_

_LR Finder is complete, type {learner\_name}.recorder.plot() to see the graph._

_learner.recorder.plot()_

png

png

And now to actually train the model.

_learner.fit\_one\_cycle(config.epochs, max\_lr=config.max\_lr)_

_\&lt;div\&gt;_
 _   \&lt;style\&gt;_
 _       /\* Turns off some styling \*/_
 _       progress {_
 _           /\* gets rid of default border in Firefox and Opera. \*/_
 _           border: none;_
 _           /\* Needs to be in here for Safari polyfill so background images work as expected. \*/_
 _           background-size: auto;_
 _       }_
 _       .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {_
 _           background: #F44336;_
 _       }_
 _   \&lt;/style\&gt;_
 _ \&lt;progress value=&#39;0&#39; class=&#39;&#39; max=&#39;1&#39;, style=&#39;width:300px; height:20px; vertical-align: middle;&#39;\&gt;\&lt;/progress\&gt;_
 _ 0.00% [0/1 00:00\&lt;00:00]_
_\&lt;/div\&gt;_

epoch

train\_loss

valid\_loss

time

_\&lt;div\&gt;_
 _   \&lt;style\&gt;_
 _       /\* Turns off some styling \*/_
 _       progress {_
 _           /\* gets rid of default border in Firefox and Opera. \*/_
 _           border: none;_
 _           /\* Needs to be in here for Safari polyfill so background images work as expected. \*/_
 _           background-size: auto;_
 _       }_
 _       .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {_
 _           background: #F44336;_
 _       }_
 _   \&lt;/style\&gt;_
 _ \&lt;progress value=&#39;785&#39; class=&#39;&#39; max=&#39;1312&#39;, style=&#39;width:300px; height:20px; vertical-align: middle;&#39;\&gt;\&lt;/progress\&gt;_
 _ 59.83% [785/1312 26:30\&lt;17:47 0.4952]_
_\&lt;/div\&gt;_

_help__(learner)_

See how simple that was?

# Predictions

Now to generate predictions. This is where you can get tripped up because the _databunch_ does not load data in sorted order. So we&#39;ll have to do reorder the generated predictions to match their original order.

如果您使用的是整个数据集，则需要一段时间。 训练完模型后，我们现在要生成预测。这可能有点棘手，因为预测生成仍然不如fastai的其他部分记录在案。我们必须要注意的是，数据加载器不会按排序顺序加载数据。这意味着我们必须对预测进行重新排序以匹配原始排序（此代码是从与文本相关的学习者代码中借用的）。

**def** _get\_preds\_as\_nparray(ds\_type) -\&gt; np.ndarray:_
 _   _&quot;&quot;&quot;
    the get\_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    &quot;&quot;&quot;
 _   preds = learner.get\_preds(ds\_type)[__0__].detach().cpu().numpy()_
 _   sampler = [i_ **for** _i in databunch.dl(ds\_type).sampler]_
 _   reverse\_sampler = np.argsort(sampler)_
 _   _ **return** _preds[reverse\_sampler, :]_

_test\_preds = get\_preds\_as\_nparray(DatasetType.Test)_

You can generate a submission if you like, though you&#39;ll probably want to use a different set of configurations.

# sample\_submission = pd.read\_csv(DATA\_ROOT / &quot;sample\_submission.csv&quot;)
# if config.testing: sample\_submission = sample\_submission.head(test.shape[0])
# sample\_submission[label\_cols] = test\_preds
# sample\_submission.to\_csv(&quot;predictions.csv&quot;, index=False)
