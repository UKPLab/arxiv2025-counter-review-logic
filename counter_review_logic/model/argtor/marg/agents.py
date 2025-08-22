import collections
import logging

from cerg.llms.OpenAi import OpenAiChatLLM
from langchain_core.prompts import ChatPromptTemplate

from cerg.models.argtors.marg.utils import colorify, counter_jaccard


class LlmChatBot:
    """
    From gpt_chatbot.py in marg-reviewer
    """

    def __init__(self, model_name, system_prompt=None, max_tokens=2048, local_model=None):
        self.model_name = model_name
        self.llm = None

        self.system_prompt = system_prompt

        self.max_tokens = max_tokens
        self.quiet = False
        self.name = None
        self.agent_name = None

        self.messages = []

        self.local_model = local_model

        self.setup()
        self.reset()

    def setup(self):
        if not self.local_model:
            self.llm = OpenAiChatLLM(
                f"marg-{self.name if self.name is not None else self.agent_name}",
                model=self.model_name,
                multimodal=False,
                config=dict(
                    temperature=0,
                    max_tokens=self.max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            )
        else:
            self.llm = self.local_model
            self.llm.set_config(dict(
                temperature=0,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            ))

    def reset(self):
        self.messages = []
        if self.system_prompt is not None:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def chat(self, prompt, role="user", max_tokens=None):
        if max_tokens is None:
            max_tokens = self.max_tokens

        # append prompt as newest message
        self.messages.append({"role": role, "content": prompt})
        if self.name is not None:
            self.messages[-1]["name"] = self.name

        # call llm
        self.llm.set_prompt(ChatPromptTemplate.from_messages([(m["role"], m["content"].replace("{", "{{").replace("}", "}}")) for m in self.messages]))
        result_text = self.llm({}, max_tokens=max_tokens)

        # append to own message thread
        self.messages.append(
            {"role": "ai", "content": result_text}
        )

        return result_text


class MultiAgentGroup:
    """
    From multi_agent.py in marg-reviewer
    """

    def __init__(
            self,
            config,
            doc_edits,
            model_name,
            paper_chunk_size=4 * (2 ** 10),
            max_tokens=800,
            prompts=None,
            quiet=False,
            use_history_pruning=False,
            master_chunk_type="normal",
            color_format="ansi",
            taxonomy="",
            extra_bots=None,
            raw_paper_chunks=None,
            local_model=None
    ):
        self.logger = logging.getLogger("marg")

        self.config = config
        self.doc_edits = doc_edits
        self.model_name = model_name
        self.paper_chunk_size = paper_chunk_size
        self.max_tokens = max_tokens
        self.prompts = prompts
        self.quiet = quiet
        self.use_history_pruning = use_history_pruning
        self.master_chunk_type = master_chunk_type
        self.color_format = color_format
        self.taxonomy = taxonomy
        self.extra_bots = extra_bots or []

        self.local_model = local_model

        self.paper_chunks = raw_paper_chunks
        assert raw_paper_chunks is not None, "Raw paper chunks must be provided"

        self.bot_experts = []
        self.init_swarm()

    def init_swarm(self):
        CHAT_COLORS = [
            "magenta",
            "blue",
            "yellow",
            "green",
            "cyan",
            "red",
            "strong-blue",
            "strong-yellow",
            "strong-green",
            "strong-cyan",
            "strong-red",
        ]
        self.bot_experts = []
        self.extra_bot_experts = []

        agent_idx = 0
        if self.master_chunk_type == "none":
            bot = LlmChatBot(
                model_name=self.model_name,
                system_prompt=self.prompts["master_system_prompt"],
                max_tokens=self.max_tokens,
                local_model=self.local_model
            )

            bot.paper_chunk = ""
            bot.agent_type = "leader"
            bot.agent_name = "Agent {}".format(agent_idx)
            bot.chat_color = CHAT_COLORS[agent_idx % len(CHAT_COLORS)]

            self.bot_experts.append(bot)

        for idx, chunk in enumerate(self.paper_chunks):
            agent_idx = len(self.bot_experts)
            if agent_idx == 0:
                bot = LlmChatBot(
                    model_name=self.model_name,
                    system_prompt=self.prompts["master_system_prompt"],
                    max_tokens=self.max_tokens,
                    local_model=self.local_model
                )
                bot.agent_type = "leader"
            else:
                bot = LlmChatBot(
                    model_name=self.model_name,
                    system_prompt=self.prompts["worker_system_prompt"],
                    max_tokens=self.max_tokens,
                    local_model=self.local_model
                )
                bot.agent_type = "worker"

            bot.paper_chunk = chunk
            bot.agent_name = "Agent {}".format(agent_idx)
            bot.chat_color = CHAT_COLORS[agent_idx]
            self.bot_experts.append(bot)

        for _, bot_prompts in enumerate(self.extra_bots):
            agent_idx = len(self.bot_experts)
            bot = LlmChatBot(
                model_name=self.model_name,
                system_prompt=bot_prompts["system_prompt"],
                max_tokens=self.max_tokens,
                local_model=self.local_model
            )
            bot.paper_chunk = ""
            bot.agent_type = "extra"
            bot.extra_prompts = bot_prompts
            bot.agent_name = "Agent {}".format(agent_idx)
            bot.chat_color = CHAT_COLORS[agent_idx]
            self.extra_bot_experts.append(bot)
            self.bot_experts.append(bot)

        for idx, bot in enumerate(self.bot_experts):
            other_agents = [x.agent_name for x in self.bot_experts if x.agent_name != bot.agent_name]
            if bot.agent_type == "leader":
                resp0 = bot.chat(
                    self.prompts["master_chunk_prompt"].format(
                        source_paper_chunk=bot.paper_chunk,
                        num_agents=len(self.bot_experts),
                        agent_name=bot.agent_name,
                        other_agent_names=", ".join(other_agents),
                        taxonomy=self.taxonomy,
                    ),
                    max_tokens=self.max_tokens,
                )
            elif bot.agent_type == "worker":
                resp1 = bot.chat(
                    self.prompts["worker_chunk_prompt"].format(
                        source_paper_chunk=bot.paper_chunk,
                        num_agents=len(self.bot_experts),
                        agent_name=bot.agent_name,
                        other_agent_names=", ".join(other_agents),
                        taxonomy=self.taxonomy,
                    ),
                    max_tokens=self.max_tokens,
                )
            elif bot.agent_type == "extra":
                resp1 = bot.chat(
                    bot.extra_prompts["chunk_prompt"].format(
                        source_paper_chunk=bot.paper_chunk,
                        num_agents=len(self.bot_experts),
                        agent_name=bot.agent_name,
                        other_agent_names=", ".join(other_agents),
                        taxonomy=self.taxonomy,
                    ),
                    max_tokens=self.max_tokens,
                )
            else:
                raise ValueError("Unknown agent type")

    def prune_history(self, messages, skip_n=0, replace_content=None, reverse_skip_n=0):
        pruned_messages = []
        skipped_n = 0
        for msg_idx, msg in enumerate(messages):
            # Experimental: prune messages that had to be retransmitted (usually because "SEND MESSAGE" was in the wrong place)
            if (
                    msg["role"] == "assistant"
                    and (msg["content"].startswith("SEND MESSAGE:") or msg["content"].startswith("SEND FULL MESSAGE:"))
                    and msg_idx > 0
            ):
                # This looks like a retransmission; let's check by seeing if
                # the non-sent content of the previous message uses roughly the
                # same tokens
                prev_assistant_msg = None
                prev_assistant_msg_idx = None
                for prev_msg_idx, prev_msg in reversed(list(enumerate(pruned_messages))):
                    if prev_msg["role"] == "assistant" and prev_msg_idx >= len(pruned_messages) - 2:
                        prev_assistant_msg = prev_msg
                        prev_assistant_msg_idx = prev_msg_idx
                        break
                if prev_assistant_msg is not None and "SEND MESSAGE" in prev_assistant_msg["content"]:
                    prev_msg_content = prev_assistant_msg["content"][
                                       : prev_assistant_msg["content"].index("SEND MESSAGE")]
                    prev_msg_tokens = collections.Counter(prev_msg_content.split())
                    msg_tokens = collections.Counter(msg["content"].split())
                    if counter_jaccard(prev_msg_tokens, msg_tokens) > 0.9:
                        new_msg = prev_assistant_msg.copy()
                        new_msg["content"] = (
                                prev_msg_content[prev_assistant_msg["content"].index("SEND MESSAGE"):]
                                + "\n\n"
                                + msg["content"][msg["content"].index(":") + 1:]
                        )
                        while len(pruned_messages) > prev_assistant_msg_idx:
                            pruned_messages.pop()
                        pruned_messages.append(new_msg)
                        continue

            if not (msg["role"] == "system" and msg["content"].startswith("Message from Agent") and len(
                    messages) - msg_idx > reverse_skip_n):
                pruned_messages.append(msg)
                continue
            elif skipped_n < skip_n:
                skipped_n += 1
                pruned_messages.append(msg)
                continue
            elif replace_content:
                new_msg = msg.copy()
                new_msg["content"] = replace_content
                pruned_messages.append(new_msg)
                continue
        return pruned_messages

    def _self_colorprint(self, *args, **kwargs):
        # print_fn = print
        print_fn = kwargs.pop("print_fn", self.logger.info)
        if "form" in kwargs:
            if kwargs["form"] == "html":
                def print_fn(x):
                    # display(IPython.display.HTML(x.replace('\n', '<br>')))
                    self.logger.info(x.replace("\n", "<br>"))

        print_fn(colorify(*args, **kwargs))

    def ask_swarm_question(self, question, pre_prompt="", stop_strings=None):
        PRUNE_STRING = "This doesn't seem relevant to me, so I will stand by for further instructions."

        # For now we just assume bot 0 is the "main" one
        rt = self.bot_experts[0].chat("{}{}".format(pre_prompt, question))

        old_rts = {rt}
        while True:
            if not self.quiet:
                self._self_colorprint(rt, color=(self.bot_experts[0].chat_color or "red"), form=self.color_format)
            if stop_strings is not None and any(x in rt for x in stop_strings):
                break

            if "SEND FULL MESSAGE" in rt:
                msgline = rt.replace("SEND FULL MESSAGE", "")
            else:
                msgidx = rt.find("SEND MESSAGE")
                if msgidx == -1 or (msgidx > 0 and rt[msgidx - 1] not in ("\n", " ")):
                    break
                # For now we just consider one message
                msgline = rt[msgidx + len("SEND MESSAGE: "):]
            rt2s = []
            for bot_idx, bot in enumerate(self.bot_experts[1:]):
                if self.use_history_pruning:
                    if bot.agent_type == "worker":
                        # bot.messages = bot.messages[:3]
                        msgs = bot.messages
                        bot.messages = bot.messages[:5]
                        if len(msgs) >= 7:
                            bot.messages[3:5] = msgs[5:7]
                    elif bot.agent_type == "extra":
                        new_msgs = []
                        for msg_idx, msg in enumerate(bot.messages):
                            if (msg["role"] == "assistant" and msg["content"].strip() == PRUNE_STRING) or (
                                    msg_idx < len(bot.messages) - 1
                                    and bot.messages[msg_idx + 1]["content"].strip() == PRUNE_STRING
                                    and bot.messages[msg_idx + 1]["role"] == "assistant"
                            ):
                                continue
                            new_msgs.append(msg)
                        bot.messages = new_msgs
                rt2 = bot.chat("Message from Agent 0: {}".format(msgline), role="system",
                               max_tokens=self.max_tokens)
                if not self.quiet:
                    self._self_colorprint(rt2, color=(bot.chat_color or "blue"), form=self.color_format)

                if rt2.strip() == PRUNE_STRING:
                    continue
                rt2s.append({"agent": bot.agent_name,
                             "msg": "Message from {}: {}".format(bot.agent_name, rt2.replace("SEND MESSAGE: ", ""))})

            rtmsg = "\n\n".join(x["msg"] for x in rt2s)
            if rtmsg.strip() == "":
                rtmsg = "No messages were received from any agent; you may need to reword and double-check your message so that the agents know who your message is intended for."
            rtmsg += "\n\nSystem note: remember that if you need to send a message back, you need to prepend SEND MESSAGE"
            rt = self.bot_experts[0].chat(rtmsg, role="system", max_tokens=self.max_tokens)
            while rt in old_rts:
                print("DUP ERROR: {}".format(rt))
                rt2 = self.bot_experts[0].chat(
                    "Error: you tried to send exactly the same message as before.  You have already received a response to that message from all agents.  Note: If the conversation is over, you should stop sending messages.",
                    role="system",
                )
                if rt2 == rt:
                    print("SUPER DUP ERROR: {}".format(rt2))
                    return rt2
                rt = rt2

            if self.use_history_pruning:
                self.bot_experts[0].messages = self.prune_history(
                    self.bot_experts[0].messages,
                    skip_n=0,
                    replace_content="[ The agents responded to you, but their messages have been pruned from the history. Your response below was created based on their real messages. ]",
                )
            old_rts.add(rt)

        return rt
