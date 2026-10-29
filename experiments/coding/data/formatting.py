# Source: https://github.com/LeapLabTHU/Absolute-Zero-Reasoner/blob/484afa480c8f6fd77faa3d35451f24f287f58ee1/absolute_zero_reasoner/data_construction/process_data.py; modified for CompassPlay.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.reward_score.math_reward import remove_boxed, last_boxed_only_string

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

instruction_following = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {}\nAssistant: <think>"

boxed_instruction = "{}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
