構想としては文書名とtoken投げたら解析してほしい

TFIDFVectorizer にcorpusをセットして追加していけるかんじ

Corpusは
- 総ドキュメント数
- 単語と出現数のmap

TFIDFVectorizer は
- 単語とその次元のmap
- ドキュメントと(TFベクトル, 単語数)のmap