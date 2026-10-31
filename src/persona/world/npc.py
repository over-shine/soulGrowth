"""照料者（NPC）。

她不是「会说话的提示词」，而是一个**有延迟、会疲劳、概率响应**的外部世界物体：
响不响应、多久响应、说什么，都由世界状态与随机流决定，婴儿无法左右。
这正是「自主社交」的起点——对方是真实的他人，不是回声壁。
"""

from __future__ import annotations

from dataclasses import dataclass

from persona.rng import bernoulli, derive, uniform
from persona.world.objects import WorldState

#: 照料者会说的话（S1 起才被婴儿听到并进入词库）。
#: 词库的 `form` 必须是这些句子的**连续子串**，所以这里刻意用短句、重复词。
#:
#: 语料是"世界的一部分"：一个孩子能长出多少词，首先取决于他听见了多少种说法。
#: 所以这里不是随手写几句安慰话，而是**按词表造句子**——每个目标词至少出现在
#: 三种说法、两类情境里，否则它永远长不成词（同样的话只听一遍不算学会）。
_NAMES: tuple[str, ...] = (
    "妈妈",
    "爸爸",
    "奶奶",
    "宝宝",
    "手手",
    "脚脚",
    "球球",
    "积木",
    "杯子",
    "镜子",
    "摇铃",
    "书书",
    "车车",
    "花花",
    "狗狗",
    "猫猫",
    "灯灯",
    "窗窗",
    "被被",
    "水水",
    "饭饭",
    "苹果",
    "香蕉",
    "面包",
    "牛奶",
    "勺子",
    "帽子",
    "鞋鞋",
    "袜子",
    "桌子",
    "椅子",
    "门门",
    "床床",
    "枕头",
    "毛巾",
    "电话",
    "熊",
    "兔子",
    "鸭子",
    "小鸡",
    "大象",
    "猴子",
    "树叶",
    "白云",
    "太阳",
    "月亮",
    "星星",
    "蛋糕",
    "糖果",
    "饼干",
    # 家里还有别的东西。语料比「目标词汇量」大得多是常态：
    # 听见一百个词，先说得出五十个（M2.2）。
    "沙发",
    "电视",
    "钥匙",
    "口袋",
    "雨伞",
    "篮子",
    "气球",
    "风车",
    "蜡笔",
    "纸张",
    "风筝",
    "沙子",
    "石头",
    "小河",
    "蚂蚁",
    "蝴蝶",
    "青蛙",
    "老虎",
    "熊猫",
    "火车",
    "飞机",
    "轮船",
    "冰淇淋",
    "面条",
    "鸡蛋",
    "果汁",
    "葡萄",
    "橘子",
    "西瓜",
    "玉米",
    "蘑菇",
)

_REFERENCES: tuple[str, ...] = ("这个", "那个", "什么", "这里", "那里")

UTTERANCES: dict[str, tuple[str, ...]] = {
    "greet": (
        "宝宝醒啦",
        "宝宝看妈妈",
        "妈妈在这里",
        "妈妈抱宝宝",
        "宝宝笑一个",
        "早上好宝宝",
        "宝宝醒来了",
        "我是妈妈",
        "我在这里",
        "看我这里",
    ),
    "feed": (
        "宝宝饿了吗",
        "奶奶来了",
        "宝宝喝奶奶",
        "奶奶香香",
        "宝宝吃饱了",
        "吃水水",
        "吃饭饭",
        "我喂宝宝",
        "我来喂你",
        "我抱宝宝",
    ),
    "soothe": (
        "宝宝不怕",
        "妈妈抱抱",
        "不怕不怕",
        "抱抱宝宝",
        "妈妈拍拍",
        "宝宝睡觉",
        "拍拍宝宝",
        "宝宝乖乖",
        "妈妈亲亲",
        "我抱抱你",
        "我在这里",
    ),
    # 指着东西说名字：这是孩子长出「词」的主战场
    "name": tuple(f"这是{w}" for w in _NAMES),
    # 玩的时候再说一遍：同一个词在不同情境里重复，才算真的懂了
    "play": tuple(f"看{w}" for w in _NAMES)
    + tuple(f"{w}在这里" for w in _NAMES[:10])
    + (
        "我来搭",
        "看我搭",
        "我来推",
        "我拿这个",
        "我要这个",
        "我推球球",
        "我搭积木",
        "我看这里",
    ),
    # 提问用：把指代词也喂进去，'这个/什么' 才会成为词
    "ask": tuple(f"{w}在哪里" for w in _NAMES)
    + tuple(f"{r}是什么" for r in _REFERENCES)
    + ("宝宝看这个", "这个在这里", "我要那个", "我找不到", "我在这里", "我要这个"),
}

#: 哭声理由 → 照料者最常做的动作
RESPONSE_BY_REASON: dict[str, tuple[str, ...]] = {
    "hunger": ("feed", "hold"),
    "discomfort": ("hold", "soothe"),
    "startle": ("soothe", "hold"),
    "sleepy": ("soothe",),
}


@dataclass
class Response:
    kind: str  # feed | hold | soothe | none
    latency_min: float
    speech: str | None
    touch: bool


class Caregiver:
    """一个照料者的行为规则。无状态：每次调用把状态写回 WorldState。"""

    def __init__(self, oid: str = "mother", *, response_prob: float = 0.75) -> None:
        self.oid = oid
        self.response_prob = response_prob

    def respond_to_cry(
        self, world: WorldState, *, day: int, episode_id: str, reason: str, stage: str
    ) -> Response:
        """响应概率 0.75、延迟 1–8 分钟（docs/08 A0.CRY）。"""
        rng = derive("caregiver", day, episode_id, "cry-response")
        if not world.caregiver_present or not bernoulli(rng, self.response_prob):
            world.bump("cry_unanswered")
            return Response(kind="none", latency_min=0.0, speech=None, touch=False)

        kinds = RESPONSE_BY_REASON.get(reason, ("soothe",))
        kind = kinds[rng.randrange(len(kinds))]
        latency = round(uniform(rng, 1.0, 8.0), 3)
        world.caregiver_last_response_min = latency
        world.bump(f"caregiver_{kind}")

        speech = None
        if stage != "S0":  # S0 只有声音，没有语句
            pool = UTTERANCES.get(kind if kind in UTTERANCES else "soothe", ())
            if pool:
                speech = pool[rng.randrange(len(pool))]

        touch = kind in ("hold", "feed")
        self._apply(world, kind)
        return Response(kind=kind, latency_min=latency, speech=speech, touch=touch)

    def _apply(self, world: WorldState, kind: str) -> None:
        mother = world.obj(self.oid)
        if mother is not None:
            # 靠近：位置向婴儿移动，触摸通道随后能感知到「人」
            mother.pos = (0.0, -0.18, 0.30)
            mother.temperature = 0.70
        if kind == "feed":
            world.bump("fed")
        elif kind == "hold":
            world.bump("held")
        elif kind == "soothe":
            world.bump("soothed")

    def drift(self, world: WorldState, *, hours: float) -> None:
        """长时间没人哭，她会走开一会儿（在场率不是恒 1）。"""
        if world.clock_hours >= 22 or world.clock_hours < 5:
            world.caregiver_present = False
        else:
            world.caregiver_present = True
        world.caregiver_mood = min(1.0, max(0.0, world.caregiver_mood + 0.01 * hours))

    def speak(self, situation: str, *, day: int, episode_id: str) -> str | None:
        pool = UTTERANCES.get(situation)
        if not pool:
            return None
        rng = derive("caregiver-speech", day, episode_id, situation)
        return pool[rng.randrange(len(pool))]
