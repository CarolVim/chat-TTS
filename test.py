import ChatTTS
import soundfile as sf

chat = ChatTTS.Chat()
chat.load_models()
texts = ["想成为股票经纪人有两个诀窍，首先你得放松，你撸管儿吗？一周多少次？您这频率只能算是个菜鸟，我个人来讲，一天至少两次",]
wavs = chat.infer(texts, use_decoder=True)
sf.write(r"./test.wav", wavs[0][0], 24000)