import os
import logging
import numpy as np
import tiktoken
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .utils.execute_code import extract_and_run_python_code
from .utils.extractor import extract_answer, extract_cheatsheet

from dotenv import load_dotenv
load_dotenv("config.env")

from text_generation import UnifiedLLMClient

logger = logging.getLogger(__name__)

# Mapping from model_name prefixes to UnifiedLLMClient provider names
_PROVIDER_MAP = {
    "openai": "openai",
    "anthropic": "claude",
    "claude": "claude",
    "gemini": "gemini",
    "google": "gemini",
    "xai": "xai",
    "grok": "xai",
    # OpenAI-compatible providers (routed through OpenAI client with custom base_url)
    "together_ai": "openai",
    "together": "openai",
    "deepseek": "openai",
    "ollama": "openai",
    "swissai": "openai",
}

# Configuration for OpenAI-compatible providers
_OPENAI_COMPATIBLE = {
    "together_ai": {"base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "together": {"base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": None},
    "swissai": {"base_url": "https://api.swissai.svc.cscs.ch/v1", "api_key_env": "SWISSAI_API_KEY"},
}

# All supported approach names
SUPPORTED_APPROACHES = [
    "default",
    "DynamicCheatsheet_Cumulative",
    "DynamicCheatsheet_RetrievalSynthesis",
    "DynamicCheatsheet_CumulativeRetrieval",
    "FullHistoryAppending",
    "Dynamic_Retrieval",
]


class LanguageModel:
    def __init__(self,
        model_name: str,
        extra_api_params: Dict[str, Any] = None,
        container_id: str = None,
    ) -> None:
        """
        LanguageModel class to interact with different language models.

        Uses UnifiedLLMClient to support OpenAI, Anthropic/Claude, Google Gemini,
        xAI/Grok, and any OpenAI-compatible provider (Together AI, DeepSeek, Ollama, etc.).

        Model name format: "provider/model" (e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250514",
        "gemini/gemini-2.5-flash", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo").

        Arguments:
            model_name : str : The name of the language model in "provider/model" format.
            extra_api_params : Dict[str, Any] : Additional provider-specific API parameters
                that are forwarded to every generate call (e.g., {"reasoning": {"effort": "medium"}}
                for OpenAI reasoning models).

        Raises:
            ValueError : If the provider cannot be determined from the model name.
        """
        self.model_name = model_name

        # Parse provider and model from model_name
        provider_prefix, model_id = self._parse_model_name(model_name)
        self.provider_prefix = provider_prefix
        self.model_id = model_id

        # Build kwargs for UnifiedLLMClient
        provider = _PROVIDER_MAP.get(provider_prefix)
        if provider is None:
            raise ValueError(
                f"Unknown provider prefix '{provider_prefix}' in model name '{model_name}'. "
                f"Supported prefixes: {', '.join(sorted(_PROVIDER_MAP.keys()))}"
            )

        client_kwargs = {"provider": provider, "model": model_id, "max_retries": 3, "retry_delay": 10}

        # Handle OpenAI-compatible providers (custom base_url and api_key)
        if provider_prefix in _OPENAI_COMPATIBLE:
            config = _OPENAI_COMPATIBLE[provider_prefix]
            client_kwargs["base_url"] = config["base_url"]
            if config["api_key_env"]:
                api_key = os.environ.get(config["api_key_env"])
                if api_key:
                    client_kwargs["api_key"] = api_key
                else:
                    raise ValueError(
                        f"API key not found. Set the {config['api_key_env']} environment variable "
                        f"or add it to config.env."
                    )
            else:
                # No API key needed (e.g., Ollama) — pass a dummy key
                client_kwargs["api_key"] = "no-key-needed"

        self.extra_api_params = extra_api_params or {}
        self.container_id = container_id
        self.client = UnifiedLLMClient(**client_kwargs)
        self.gpt4Tokenizer = tiktoken.encoding_for_model('gpt-4o')

    @staticmethod
    def _parse_model_name(model_name: str) -> Tuple[str, str]:
        """
        Parse "provider/model" format. For providers like together_ai where the
        model part itself contains slashes (e.g., "together_ai/meta-llama/Llama-3.3-70B"),
        only the first segment is treated as the provider prefix.

        If no "/" is present, the provider is inferred from the model name.

        Returns:
            (provider_prefix, model_id)
        """
        if "/" in model_name:
            prefix, rest = model_name.split("/", 1)
            return prefix, rest

        # Infer provider from model name when no prefix is given
        if model_name.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai", model_name
        elif model_name.startswith("claude-"):
            return "anthropic", model_name
        elif model_name.startswith("gemini-"):
            return "gemini", model_name
        elif model_name.startswith("grok-"):
            return "xai", model_name
        else:
            raise ValueError(
                f"Cannot infer provider from model name '{model_name}'. "
                f"Use 'provider/model' format (e.g., 'openai/gpt-4o')."
            )

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text using the GPT-4o tokenizer (approximate).
        """
        tokens = self.gpt4Tokenizer.encode(text)
        return len(tokens)

    # Sentinel value used when a provider handles code execution natively
    # without requiring a container (e.g. Claude's code_execution tool).
    _NATIVE_CODE_EXECUTION_ID = "native"

    # Code execution output longer than this is truncated before it is fed back to the model
    MAX_CODE_OUTPUT_CHARS = 8000

    def create_container(self, **kwargs) -> str:
        """
        Create a code interpreter session and return an identifier.

        For OpenAI models this creates a server-side container via the Responses API.
        For Claude models no container is needed — the code execution tool is
        enabled per-request, so a sentinel ID is returned immediately.
        """
        if self.provider_prefix in ("claude", "anthropic"):
            self.container_id = self._NATIVE_CODE_EXECUTION_ID
            logger.info("Claude native code execution enabled (no container required).")
            return self.container_id

        container_id = self.client.create_container(**kwargs)
        self.container_id = container_id
        return container_id

    def generate(self,
        history: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        current_depth: int = 1,
        max_depth_num_rounds: int = 3,
        allow_code_execution: bool = True,
        code_execution_flag: str = "EXECUTE CODE!",
        final_output: str = "",
        use_code_interpreter: bool = False,
    ) -> str:
        """
        Generate a response from the language model, with optional iterative code execution.

        When allow_code_execution is True and the model produces a Python code block
        followed by the code_execution_flag, the code is executed and its output is
        appended to the conversation. The model is then prompted to continue, up to
        max_depth_num_rounds iterations.

        Arguments:
            history : List[Dict[str, str]] : The conversation history (list of message dicts).
            temperature : float : The sampling temperature for the model.
            max_tokens : int : The maximum number of tokens to generate.
            current_depth : int : The current iteration depth (for recursive code execution).
            max_depth_num_rounds : int : The maximum number of code execution rounds allowed.
            allow_code_execution : bool : Whether to allow Python code execution.
            code_execution_flag : str : The trigger string for code execution.
            final_output : str : Accumulated output across recursive calls.

        Returns:
            str : The final accumulated output.

        Raises:
            ValueError : If the history is empty.
        """
        if len(history) == 0:
            raise ValueError("History must contain at least one message.")

        # Code interpreter path: handled server-side by the provider
        if use_code_interpreter and self.container_id:
            if self.provider_prefix in ("claude", "anthropic"):
                # Claude: native code execution tool (no container required)
                output = self.client.generate_with_code_interpreter_claude(
                    messages=history,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    **self.extra_api_params,
                )
            else:
                # OpenAI: Responses API with a persistent container
                output = self.client.generate_with_code_interpreter(
                    messages=history,
                    container_id=self.container_id,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    **self.extra_api_params,
                )
            if not output:
                logger.warning("Received empty response from code interpreter.")
                output = "(No response generated)"
            return f"{final_output}\n\n{output}".strip() if final_output else output

        # Generate the response from the language model
        output = self.client.generate_from_messages(
            messages=history,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **self.extra_api_params,
        )

        # Guard against empty or None responses
        if not output:
            logger.warning("Received empty response from the language model.")
            output = "(No response generated)"

        # Check if the model produced a code block followed by the execution flag
        if allow_code_execution and code_execution_flag in output:
            pre_flag_text = output.split(code_execution_flag)[0].strip()
            # Only execute if there's a code block ending right before the flag
            if len(pre_flag_text) >= 3 and pre_flag_text.endswith("```"):
                output_prefix = pre_flag_text
                executed_code = extract_and_run_python_code(output_prefix)
                if executed_code:
                    executed_code = executed_code.strip()
                    # Huge prints (e.g. a loop dumping candidates) can overflow the context window on the next round
                    if len(executed_code) > self.MAX_CODE_OUTPUT_CHARS:
                        head, tail = executed_code[:6000], executed_code[-2000:]
                        executed_code = f"{head}\n... [{len(executed_code) - 8000} characters of output truncated] ...\n{tail}"
                else:
                    executed_code = "(No code block found to execute)"
                current_output = f"{output_prefix}\n{code_execution_flag}\n\n{executed_code}"
                final_output = f"{final_output}\n\n{current_output}".strip()

                # If we haven't exceeded the max depth, prompt the model to continue
                if current_depth <= max_depth_num_rounds:
                    warning_txt = ""
                    if current_depth == max_depth_num_rounds:
                        warning_txt = " (This is the last round. No more code execution will be allowed. Please present your final solution now.)"
                    new_messages = [
                        {"role": "assistant", "content": current_output},
                        {"role": "user", "content": f"Proceed with any additional steps required and provide the completed solution. If everything is already complete, type FINAL ANSWER and submit it in the expected format. If you are stuck, please try alternative methods to solve the problem and provide the final solution.{warning_txt}"}
                    ]
                    history += new_messages
                    return self.generate(
                        history=history,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        current_depth=current_depth + 1,
                        max_depth_num_rounds=max_depth_num_rounds,
                        allow_code_execution=allow_code_execution,
                        code_execution_flag=code_execution_flag,
                        final_output=final_output,
                    )
                else:
                    return f"{final_output}\n\n{current_output}".strip()

        # No code execution — just append output
        final_output = f"{final_output}\n\n{output}".strip()
        return final_output

    def advanced_generate(self,
        approach_name: str,
        input_txt: str,
        cheatsheet: str = None,
        generator_template: str = None,
        cheatsheet_template: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_num_rounds: int = 1,
        allow_code_execution: bool = True,
        code_execution_flag: str = "EXECUTE CODE!",
        add_previous_answers_to_cheatsheet: bool = True,
        original_input_corpus: List[str] = None,
        original_input_embeddings: np.ndarray = None,
        generator_outputs_so_far: List[str] = None,
        retrieve_top_k: int = 3,
        use_code_interpreter: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a response using one of the supported Dynamic Cheatsheet approaches.

        Supported approaches:
            - "default": Single LLM call, no cheatsheet.
            - "DynamicCheatsheet_Cumulative": Maintains and grows a persistent cheatsheet
              across queries. After each answer, a curator call updates the cheatsheet.
            - "DynamicCheatsheet_RetrievalSynthesis": Retrieves top-k similar past examples,
              synthesizes them into a custom cheatsheet, then generates an answer.
            - "DynamicCheatsheet_CumulativeRetrieval": Hybrid — maintains a cumulative cheatsheet
              AND retrieves top-k similar examples. The generator receives both general strategies
              (from the cumulative cheatsheet) and specific examples (from retrieval).
            - "FullHistoryAppending": Appends all previous input-output pairs (no curation).
            - "Dynamic_Retrieval": Retrieves top-k similar examples without synthesis.

        Arguments:
            approach_name : str : The name of the approach to use.
            input_txt : str : The input text for the model.
            cheatsheet : str : The current cheatsheet content.
            generator_template : str : The template for the generator model.
            cheatsheet_template : str : The template for the cheatsheet curator model.
            temperature : float : The sampling temperature for the model.
            max_tokens : int : The maximum number of tokens to generate.
            max_num_rounds : int : The maximum number of refinement rounds.
            allow_code_execution : bool : Whether to allow Python code execution.
            code_execution_flag : str : The flag to trigger code execution.
            add_previous_answers_to_cheatsheet : bool : Whether to include previous answers in the cheatsheet.
            original_input_corpus : List[str] : The original input texts (for retrieval-based approaches).
            original_input_embeddings : np.ndarray : Pre-computed embeddings (for retrieval-based approaches).
            generator_outputs_so_far : List[str] : Previous generator outputs.
            retrieve_top_k : int : Number of similar examples to retrieve.

        Returns:
            Dict containing: input_txt, steps, final_answer, final_output, final_cheatsheet, etc.

        Raises:
            ValueError : If the approach name is not recognized or required templates are missing.
        """

        # When using code interpreter, disable local subprocess execution
        if use_code_interpreter:
            allow_code_execution = False

        if approach_name == "default":
            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", "(empty)")
            generator_history = [
                {"role": "user", "content": generator_prompt},
            ]
            generator_output = self.generate(
                history=generator_history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
                use_code_interpreter=use_code_interpreter,
            )

            generator_answer = extract_answer(generator_output)

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": None,
                        "new_cheatsheet": None,
                    }
                ],
                "previous_answers": None,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": None,
                "generator_output": generator_output,
            }

        elif approach_name == "DynamicCheatsheet_Cumulative":
            if cheatsheet is None:
                raise ValueError("Cheatsheet must be provided for DynamicCheatsheet_Cumulative approach.")
            if cheatsheet_template is None:
                raise ValueError("Cheatsheet template must be provided for DynamicCheatsheet_Cumulative approach.")

            steps = []
            previous_answers = []
            generator_output = ''

            for round_idx in range(max(1, max_num_rounds)):
                ## STEP 1: Run the generator model with the input text and the cheatsheet
                generator_cheatsheet_content = cheatsheet

                # If there are previous answers, add them to the cheatsheet content for the generator
                if round_idx > 0 and add_previous_answers_to_cheatsheet:
                    previous_answers_txt = f"PREVIOUS ANSWERS:\n{'; '.join(previous_answers)}"
                    generator_cheatsheet_content = f"{generator_cheatsheet_content}\n\n{previous_answers_txt}"

                generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", generator_cheatsheet_content)
                current_cheatsheet = cheatsheet

                # Prepare the message history for the generator model
                generator_history = [{"role": "user", "content": generator_prompt}]
                # Run the generator model
                generator_output = self.generate(
                    history=generator_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                    use_code_interpreter=use_code_interpreter,
                )
                # Extract the output from the generator model
                generator_answer = extract_answer(generator_output)

                ## STEP 2: Run the cheatsheet extraction model with the generator output and the current cheatsheet
                cheatsheet_prompt = cheatsheet_template.replace("[[QUESTION]]", input_txt).replace("[[MODEL_ANSWER]]", generator_output).replace("[[PREVIOUS_CHEATSHEET]]", current_cheatsheet)

                cheatsheet_history = [{"role": "user", "content": cheatsheet_prompt}]
                cheatsheet_output = self.generate(
                    history=cheatsheet_history,
                    temperature=temperature,
                    max_tokens=2 * max_tokens,
                    allow_code_execution=False,
                )

                # Extract the new cheatsheet from the output (if present); otherwise, return the old cheatsheet
                new_cheatsheet = extract_cheatsheet(response=cheatsheet_output, old_cheatsheet=current_cheatsheet)
                cheatsheet = new_cheatsheet

                previous_answers.append(f"Round {round_idx+1}: {generator_answer}")

                steps.append({
                    "round": round_idx,
                    "generator_prompt": generator_prompt,
                    "generator_output": generator_output,
                    "generator_answer": generator_answer,
                    "current_cheatsheet": current_cheatsheet,
                    "new_cheatsheet": new_cheatsheet,
                })

            return {
                "input_txt": input_txt,
                "steps": steps,
                "previous_answers": previous_answers,
                "final_answer": generator_answer,
                "final_cheatsheet": new_cheatsheet,
                "final_output": generator_output,
            }

        elif approach_name == "FullHistoryAppending":
            length_of_history = len(generator_outputs_so_far)
            if length_of_history > 0:
                top_k_original_inputs = original_input_corpus[:length_of_history]
                top_k_original_outputs = generator_outputs_so_far

                curated_cheatsheet = "### PREVIOUS SOLUTIONS (START)\n\n"
                for i, (previous_input_txt, previous_output_txt) in enumerate(zip(original_input_corpus, generator_outputs_so_far)):
                    curated_cheatsheet += f"#### Previous Input #{i+1}:\n\n{previous_input_txt}\n\n#### Model Solution to Previous Input #{i+1}:\n\n{previous_output_txt}\n---\n---\n\n"
                curated_cheatsheet += "#### PREVIOUS SOLUTIONS (END)"
            else:
                top_k_original_inputs = []
                top_k_original_outputs = []
                curated_cheatsheet = "(empty)"

            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", curated_cheatsheet)
            generator_history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                history=generator_history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
                use_code_interpreter=use_code_interpreter,
            )
            generator_answer = extract_answer(generator_output)

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": curated_cheatsheet,
                        "new_cheatsheet": None,
                    }
                ],
                "top_k_original_inputs": top_k_original_inputs,
                "top_k_original_outputs": top_k_original_outputs,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": curated_cheatsheet,
            }

        elif approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis"]:
            # Get the current original input embedding
            current_original_input_embedding = original_input_embeddings[-1]
            prev_original_input_embeddings = original_input_embeddings[:-1]

            # Retrieve the most similar k input-output pairs from the previous inputs and outputs
            if len(prev_original_input_embeddings) > 0:
                similarities = cosine_similarity([current_original_input_embedding], prev_original_input_embeddings)
                top_k_indices = np.argsort(similarities[0])[::-1][:retrieve_top_k]
                top_k_original_inputs = [original_input_corpus[i] for i in top_k_indices]
                top_k_original_outputs = [generator_outputs_so_far[i] for i in top_k_indices]
                top_k_similar_values = similarities[0][top_k_indices]
                curated_cheatsheet = "### PREVIOUS SOLUTIONS (START)\n\nNote: The input-output pairs listed below are taken from previous test cases and are meant to assist you in understanding potential solution strategies or tool usages. While they can offer insight and inspiration, they should not be blindly copied, as they may contain errors or may not fit your specific use case. Approach them with a critical mindset—analyze their logic, verify their correctness, and adapt them as needed. Your goal should be to develop a well-reasoned solution that best addresses the problem at hand.\n\n"
            else:
                top_k_original_inputs = []
                top_k_original_outputs = []
                top_k_similar_values = []
                curated_cheatsheet = '(empty)'

            # Add the previous input-output pairs to the cheatsheet (reverse order so most similar is last/closest to the question)
            for i, (previous_input_txt, previous_output_txt, similarity) in enumerate(zip(top_k_original_inputs[::-1], top_k_original_outputs[::-1], top_k_similar_values[::-1])):
                curated_cheatsheet += f"#### Previous Input #{i+1} (Similarity: {similarity:.2f}):\n\n{previous_input_txt}\n\n#### Model Solution to Previous Input  #{i+1}:\n\n{previous_output_txt}\n---\n---\n\n"
            curated_cheatsheet = curated_cheatsheet.strip()

            if curated_cheatsheet != '(empty)':
                curated_cheatsheet += "\n\n#### PREVIOUS SOLUTIONS (END)"

            # For RetrievalSynthesis, run the curator to synthesize a better cheatsheet
            previous_cheatsheet = cheatsheet
            if approach_name == "DynamicCheatsheet_RetrievalSynthesis":
                cheatsheet_prompt = cheatsheet_template.replace("[[PREVIOUS_INPUT_OUTPUT_PAIRS]]", curated_cheatsheet)
                cheatsheet_prompt = cheatsheet_prompt.replace("[[NEXT_INPUT]]", input_txt)
                cheatsheet_prompt = cheatsheet_prompt.replace("[[PREVIOUS_CHEATSHEET]]", previous_cheatsheet)
                cheatsheet_history = [{"role": "user", "content": cheatsheet_prompt}]
                cheatsheet_output = self.generate(
                    history=cheatsheet_history,
                    temperature=temperature,
                    max_tokens=2 * max_tokens,
                    allow_code_execution=False,
                )
                new_cheatsheet = extract_cheatsheet(response=cheatsheet_output, old_cheatsheet=curated_cheatsheet)
                curated_cheatsheet = new_cheatsheet

            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", curated_cheatsheet)
            generator_history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                history=generator_history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
                use_code_interpreter=use_code_interpreter,
            )
            generator_answer = extract_answer(generator_output)

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": curated_cheatsheet,
                        "new_cheatsheet": None,
                    }
                ],
                "top_k_original_inputs": top_k_original_inputs,
                "top_k_original_outputs": top_k_original_outputs,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": curated_cheatsheet,
            }

        elif approach_name == "DynamicCheatsheet_CumulativeRetrieval":
            # Hybrid approach: cumulative cheatsheet + retrieval of similar examples.
            # The generator receives BOTH the evolving cheatsheet (general strategies)
            # AND the top-k most similar previous examples (task-specific context).
            # After answering, the curator updates the cumulative cheatsheet.

            if cheatsheet is None:
                raise ValueError("Cheatsheet must be provided for DynamicCheatsheet_CumulativeRetrieval approach.")
            if cheatsheet_template is None:
                raise ValueError("Cheatsheet template must be provided for DynamicCheatsheet_CumulativeRetrieval approach.")

            # --- Retrieval step: find similar previous examples ---
            retrieved_section = ""
            top_k_original_inputs = []
            top_k_original_outputs = []

            if original_input_embeddings is not None and len(original_input_embeddings) > 1:
                current_embedding = original_input_embeddings[-1]
                prev_embeddings = original_input_embeddings[:-1]

                similarities = cosine_similarity([current_embedding], prev_embeddings)
                top_k_indices = np.argsort(similarities[0])[::-1][:retrieve_top_k]
                top_k_original_inputs = [original_input_corpus[i] for i in top_k_indices]
                top_k_original_outputs = [generator_outputs_so_far[i] for i in top_k_indices]
                top_k_similar_values = similarities[0][top_k_indices]

                retrieved_section = "\n\n### RETRIEVED SIMILAR EXAMPLES (START)\n\nNote: These are the most similar previous problems and their solutions. Use them as reference, but verify correctness and adapt as needed.\n\n"
                for i, (prev_input, prev_output, sim) in enumerate(zip(
                    top_k_original_inputs[::-1], top_k_original_outputs[::-1], top_k_similar_values[::-1]
                )):
                    retrieved_section += f"#### Similar Example #{i+1} (Similarity: {sim:.2f}):\n\n{prev_input}\n\n#### Solution:\n\n{prev_output}\n---\n---\n\n"
                retrieved_section += "### RETRIEVED SIMILAR EXAMPLES (END)"

            # --- Build combined cheatsheet for the generator ---
            combined_cheatsheet = cheatsheet
            if retrieved_section:
                combined_cheatsheet = f"{cheatsheet}\n{retrieved_section}"

            # --- Generator step ---
            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", combined_cheatsheet)
            current_cheatsheet = cheatsheet

            generator_history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                history=generator_history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
                use_code_interpreter=use_code_interpreter,
            )
            generator_answer = extract_answer(generator_output)

            # --- Curator step: update the cumulative cheatsheet ---
            cheatsheet_prompt = cheatsheet_template.replace("[[QUESTION]]", input_txt).replace("[[MODEL_ANSWER]]", generator_output).replace("[[PREVIOUS_CHEATSHEET]]", current_cheatsheet)

            cheatsheet_history = [{"role": "user", "content": cheatsheet_prompt}]
            cheatsheet_output = self.generate(
                history=cheatsheet_history,
                temperature=temperature,
                max_tokens=2 * max_tokens,
                allow_code_execution=False,
            )

            new_cheatsheet = extract_cheatsheet(response=cheatsheet_output, old_cheatsheet=current_cheatsheet)

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": current_cheatsheet,
                        "new_cheatsheet": new_cheatsheet,
                        "retrieved_examples": retrieved_section,
                    }
                ],
                "top_k_original_inputs": top_k_original_inputs,
                "top_k_original_outputs": top_k_original_outputs,
                "previous_answers": None,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": new_cheatsheet,
            }

        else:
            raise ValueError(
                f"Approach '{approach_name}' not found. "
                f"Supported approaches: {', '.join(SUPPORTED_APPROACHES)}"
            )
