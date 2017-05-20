from classifiers.models.model import Model


class EmoticonExt(Model):
    def __init__(self, name="Emoticon Extended"):
        Model.__init__(self, name=name)
        self.emoticons = {
            "positive": [":)",
                         ":-)",
                         ": )",
                         ":D",
                         "=D",
                         ":-]",
                         ":]",
                         ":-3",
                         ":3",
                         ":->",
                         ":>",
                         "8-)",
                         "8)",
                         ":-}",
                         ":}",
                         ":o)",
                         ":c)",
                         ":^)",
                         "=]",
                         "=)",
                         ":‑D",
                         "8‑D",
                         "8D",
                         "x‑D",
                         "xD",
                         "X‑D",
                         "XD",
                         "=D",
                         "=3",
                         "B^D",
                         ":-))",
                         ":'‑)",
                         ":')",
                         ":-*",
                         ":*",
                         ":×",
                         ";‑)",
                         ";)",
                         "*-)",
                         "*)",
                         ";‑]",
                         ";]",
                         ";^)",
                         ":‑,",
                         ";D",
                         "<3",
                         "😁",
                         "😀",
                         "😍",
                         "😻",
                         "🤗",
                         "😇",
                         "😂",
                         "😹",
                         "😗",
                         "😽",
                         "😚",
                         "😘",
                         "😙",
                         "😆",
                         "☺️",
                         "😌",
                         "🤣",
                         "🙂",
                         "😄",
                         "😸",
                         "😃",
                         "😺",
                         "😏",
                         "😼",
                         "😛",
                         "😎",
                         "😅",
                         "😉",
                         "😋"],
            "negative": [":(",
                         ":-(",
                         ": (",
                         ":'(",
                         ":‑(",
                         ":(",
                         ":‑c",
                         ":c",
                         ":‑<",
                         ":<",
                         ":‑[",
                         ":[",
                         ":-||",
                         ">:[",
                         ":{",
                         ":@",
                         ">:(",
                         "D‑':",
                         "D:<",
                         "D:",
                         "D8",
                         "D;",
                         "D=",
                         "DX",
                         ":‑/",
                         ":/",
                         ":‑.",
                         ">:\\",
                         ">:/",
                         ":\\",
                         "=/",
                         "=\\",
                         ":L",
                         "=L",
                         ":S",
                         "</3",
                         "<\\3",
                         ">.<",
                         "v.v",
                         "😠",
                         "😧",
                         "😲",
                         "😊",
                         "🤡",
                         "😰",
                         "😖",
                         "😕",
                         "😢",
                         "😿",
                         "😞",
                         "😥",
                         "😵",
                         "🤕",
                         "🤒",
                         "😨",
                         "😳",
                         "😦",
                         "😬",
                         "😯",
                         "👿",
                         "🤥",
                         "😷",
                         "🤢",
                         "🤓",
                         "😮",
                         "😔",
                         "😣",
                         "😾",
                         "😡",
                         "🙄",
                         "😱",
                         "🙁",
                         "🙀",
                         "😈",
                         "🤧",
                         "😭",
                         "😜",
                         "😓",
                         "🤔",
                         "😫",
                         "😒",
                         "😩",
                         "😟",
                         "🤐"]
        }

    def contains_emoticon(self, sentiment, tweet):
        """
        Checks if a tweet contains an emoticon of given sentiment
        :param sentiment: String, either "positive", "negative" or "neutral"
        :param tweet: A tweet text string
        :return: boolean
        """
        for emoticon in self.emoticons[sentiment]:
            if emoticon in tweet:
                return True

        return False

    def classify(self, tweet):

        # Calculate sentiment scores
        contains_positive = self.contains_emoticon("positive", tweet)
        contains_negative = self.contains_emoticon("negative", tweet)

        if contains_positive and contains_negative:
            return 0
        if contains_positive:
            return 1
        if contains_negative:
            return -1

        return 0

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = list(map(self.classify, tweets))
        return self.predictions
