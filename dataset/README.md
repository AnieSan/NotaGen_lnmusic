## 前置处理步骤
1. MuseScore dataset 转成 ABC 格式		-> 03_abc
   1. L 统一为 1/8
   2. 不输出换行符（$）
   3. 不使用 U:s=!stemless!，直接在乐谱里打 !stemless!

   检查：
   - 全部用music21过一遍，检查有Warning的曲子（速度过慢，放弃）
   - %%MIDI 行的信息种类：由xml直接转abc的谱子无MIDI行
   - K:none 的乐器

3. 全部 unidecode 一遍				-> 04_abc_unidecoded
4. -> 05_abc_cleaned
   1. 处理 information field 和 %% 行
   - 保留 information field：L，M，K，Q，V，I
   - 清除 information field：X，T，C，W，w，Z
   - 保留 %% 行：%%score
   2. 处理不当的表示
   - 去掉连续换行符
   - 删掉行末小节号
   - 删掉 \"
   - 删掉含小节线的引号文本
   3. 去掉头尾空白小节（strip），同时删掉小节数对不上的曲子
   4. 删掉小于8小节的曲子
5. text annotation 处理				-> 06_abc_text-filtered
   1. 按空格分词统计词频，先过滤一次
   2. 将字符串划分为单词，每个单词均高于一定出现次数才保留
   3. 删掉纯数字，删掉过长的字符串，删掉""里什么都没有的字符串
   4. 处理连续的标点符号和空格，替换为单个字符
6. 转置（rotate），且检查所有小节各声部的时值，不一致则滤掉 -> 07_abc_rotated_CLAMP

   这一步同时确定 Symbolic 的数据范围，Symbolic 数据不做转置，直接走第6步

7. 去重

   第一阶段

   1. 按长度缩小对比范围，长度相差太远的不比
   2. 满足1的检查配器，配器不一样的不比
   3. 满足2的抽前8小节作为sample进行ld_sim对比，超过阈值的配对需要去重。倾向是保留信息更多的（也就是长度更长的），这里用未过滤annotation（感叹号和双引号括起来的文本）的版本进行判断

   第二阶段

   1. 在第一阶段基础上，按配器划分曲子
   2. 对于配器相同的曲子，进行metadata匹配（譬如用ld_sim），超过阈值的曲子也算重，取舍规则和第一阶段一样

## Online 数据处理

7. 移调
8. 转置							



