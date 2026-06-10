# 001 — 目標と論点マップ(v3, 2026-06-10 全面改訂)

v2 からの変更: 「Helios の実装に準拠」を撤回。**DMD2 をスクラッチから設計する**前提で、
論点を3経路(手法の構成要素 / 故障モード / 決定依存順)から独立に再導出し統合した。
Helios を含む全先行レシピは §4 の行列の1行として等価に扱う。教師プロファイル実測(§5)を反映し、
旧版の誤推定(固定費 ~25s)を訂正した。

## 1. 目標と決定事項

**Pixal3D を教師とした DMD2 蒸留により、few-step の画像→3D(geometry)生成器を作る。**

決定済み:
- **分布マッチング蒸留を主軸に、DMD2 を基準骨格とする**。ただし (a) **特定の先行実装には
  準拠しない** — 構成要素ごとに §2 の軸上で選択し、§4 の検証済みレシピ行列を判断材料とする;
  (b) コアの置換・併用(G6: VM/VD、MMD には単独成立の前例あり)は**開いた軸として残す**。
- 運用規則(ユーザ既定、設計軸ではない): DMD 収束は毎 run、decoded probe で検証する。
  surrogate 改善+品質劣化は collapse と判定する。
- 教師 = **Pixal3D / TRELLIS.2**(Pixal3D = TRELLIS.2 backbone + ProjectAttention。
  VAE/latent 空間は TRELLIS.2 凍結流用で共通)。
- 対象 = geometry(S1 sparse-structure + S2a shape-LR + S2b shape-HR)。**texture(S3)はスコープ外**
  (ただしフルパイプのデモでは生徒 geometry が教師 S3 への入力分布シフトになる点は §2-G10 の評価論点)。
- **計測・報告規約**: 最細粒度ブロック単位で計測し、報告では chunk 集計を併記する(§5 が実例)。

以下、§2-§3 は論点の地図であり解は決めない。各軸のタグ: **[T0-T5]** = 決定依存順の階層
(T0 が根、下流はその決定なしにコード化できない)、**E高/E低** = 参照群のエントロピー
(高=先行が割れている/沈黙、低=先行が一致 — 一致でも我々の設定で有効とは限らない)、
**構造的/設定級** = 後から覆すコスト。

注意 2 点: (1) **「M0」と記す根拠は全て異教師(Hunyuan3D-2.1 vecset・CFG 蒸留済み・単段)での
データ点**であり、本教師(sparse・live-CFG・多段)への移転は各軸で未検証。
(2) 選択肢の列挙順は引用密度順(検証情報の多い順)であり、選好順ではない。

## 2. 設計軸マップ(スクラッチ再導出)

### G0. 戦略ルート(T0 — 全ての下流を規定)

**A1. stage-scope** [T0, E高, 構造的] — どの段を few-step 化するか。
選択肢: HR のみ / shape 両段 / 全3段 / 1段ずつ拡張 / LR+HR を1生徒に折り畳み(パターンD)。
根拠: flow 実測 SS 1.10s / LR 1.12s / HR 10.8s(§5)— HR が flow の過半。S1 は dense 固定 4096
tokens で M0 機構が直訳、SLAT 段は varlen で書き直し。異種マルチモデル・離散 support 結合の
カスケード蒸留は検証済み先行ゼロ(§4)。

**A2. teacher-choice** [T0, E高, warm-start は構造的] — Pixal3D 単独か、TRELLIS.2 を
(a) 機構検証台 (b) 生徒 warm-start 元 として使うか。重み比較(ft 系譜確定)を先に行うか。
根拠: PyramidalWan が教師選択を明示比較した唯一の検証例で、**非カスケード元教師=定量最良 /
カスケード化教師=視覚最良の割れた結果**(§4)。「TRELLIS.2 蒸留→後から proj 追加」は
x0 分布の view-aligned 化を伴うため字面では不成立(v2 結論、維持)。

**A3. convention-gates** [T0, E低, 一度決めたら固定] — flow 規約の移植方式とゲートテスト。
選択肢: σ=1−t + v 符号反転で M0 代数を写像 / Pixal3D 規約でネイティブ再導出。ゲート
(x0-recovery、dm-grad-sign、velocity-score crosscheck、teacher self-consistency)をどこまで
走行前提条件にするか。根拠: M0 の決定登録簿は 97 決定中 47 を run-breaker と分類。規約罠
(scale_noise / add_noise の鏡像)は実際に実行可能ゲートだけが捕まえた。

### G1. 生成器(student)

**B1. step 数(NFE)/段** [T1, E中低, 設定級(1-step 選択は ODE-init パイプラインを誘発)] —
1 / 4 / 段ごと非対称(HR に厚く)/ 多→少の漸進。検証済みレシピは 4 前後に集積
(SF 4, SF++ 4, SwD 4-6, RF 5, PyramidalWan 2-2-1, CausVid 50→4)。NFE は速度の直接レバー(§5)。

**B2. anchor 配置と rescale_t** [T1, E高(warp の前例なし)] — anchor を線形 σ 格子に置くか、
教師の rescale_t ワープ格子(SS 5.0 / shape 3.0)に置くか、guidance_interval 境界(t=0.6)を
意識した配置にするか。DMD2 原典の 999/749/499/249 は教師推論スケジュールではなく一様梯子。
M0 は教師スケジューラ由来(shift=1.0)を採り、Helios の [1000,750,500,250] と実際に異なった。

**B3. パラメタ化** [T1, E低] — 教師の v 予測+時間条件付けを保持(全先行)/x̂0 ヘッド/
時間条件除去(DMD1 の 1-step 設計)。1-step 時の条件付け t も参照間で割れる
(DMD1 は T−1、DMD2 SDXL 1-step は t=399)。t=1=noise 規約と σ_min=1e-5 境界での x̂0 復元精度は
M0 で run-breaker 級(倍精度で 4.8e-07 まで検証済みの代数を移植するか再導出するか)。

**B4. backward simulation** [T2] — 中間 step 入力の作り方: DMD2 流の確率的再ノイズ
rollout / 教師推論と同じ決定論的 Euler rollout / シミュレーションなし(DMD2 が**彼らの SDXL
設定で**悪化を実証した baseline)。サブ選択: 損失を入れる rollout 位置(ランダム exit / 最終)、
exit・再ノイズ t の分布(一様 / 重み付き — Helios は Beta(4,1.5)→500 step で一様へ減衰)、
grad 範囲(過去 step detach = 全先行の慣行 / 全 rollout 逆伝播)。教師が決定論的 Euler なので
確率的 vs 決定論的再ノイズは本物の分岐。

**B5. 学習対象範囲** [T1, E中(本物の分裂点)] — full finetune(DMD2 論文)/ LoRA
(Helios r256、SwD r64-128、M0 r128; rank/target はサブ軸)/ 部分解凍。1段あたり生徒+critic+
凍結教師 ≈ 1.3B×3 + optimizer 状態がメモリ envelope を決める(G8 と連動)。

**B6. 初期化** [T2, E中高] — 教師重みコピーのみ(全先行の基底)/ ODE-init(1-step では
official DMD2 が必須化: 10k pairs, lr 1e-5; CausVid は 1000 pairs で足りた)/ 既存蒸留 ckpt
warm-start / TRELLIS.2 重み warm-start(A2 依存)。**Causal Forcing の定理**: 生徒と因子分解の
異なる教師からの ODE-init は条件付き期待値(ぼけ)解に収束 — 我々の教師は段ごとに因子分解済み
なので、段ごと ODE-init はこの病理を構造的に回避(多段教師の利点候補、未検証)。
Causal Forcing の検証済み負結果: **init の因子分解不整合は後段の DMD では修復できなかった**
(B6 を実質構造的にする最強の根拠)。注意: ODE ペアは生成時点の CFG 構成(D1/D2)を埋め込む —
ODE-init 採用時は D を先に確定しないとペア再生成コストが発生する。

**B7. EMA** [T4, E中(本物の分裂点)] — なし(official DMD2 / CausVid / SwD / RF)/
あり(Helios 0.99@750 — ただし**検証は live 重み**(use_ema_validation=false)、SF 0.99、
LongLive 0.99@200、SF++「EMA 版が良い」)。M0 では EMA は collapse の原因でも隠蔽でもなかった
(EMA≈live)。観測性の方針(live と EMA のどちらを probe・選択・公開するか)として決める。

### G2. DM コア(分布マッチング勾配)

**C1. 差分を取る空間** [T2, E低] — x̂0 空間(official: grad ∝ pred_fake − pred_real)/ ε /
v(教師ネイティブ)/ score 空間(DMD Eq.7 の書面通り)。空間と w(t) は合成して1つの実効重み —
**独立に決めると二重カウント**になる。

**C2. w(t) と normalizer、varlen 縮約** [T2, E低(式)+ **E∞(varlen は前例なし)**, 構造的] —
DMD Eq.8(per-sample L1 normalizer; σ²/α 重み)/ 単純スケジュール / なし。参照の縮約規則
「非バッチ全次元の mean」は固定形状でのみ定義 — SLAT 段では per-sample segment 縮約
(coords[:,0])/ flat token mean(大物体ほど重み増)/ ハイブリッドを明示的に決める必要。
normalizer は「誰の擬似勾配を増幅するか」を決め、scale-drift 故障モードに直結。

**C3. 勾配注入の機構** [T3, E低] — **3方式は数学的に等価な勾配注入**で、差は機構と精度のみ:
surrogate 0.5·MSE(x_gen, sg(x_gen−grad))(official/Helios)/ 直接 backprop / カスタム autograd。
どれを選んでも no_grad スコープ・detach 構造の検証ゲートは必要(M0 で run-breaker 指定だったのは
実装規則であり方式選択の根拠ではない)。精度(Helios 倍 / official 単)もサブ軸。
nan_to_num 前の生 NaN 検査を入れるか。

**C4. DM 損失の t サンプリング** [T2] — 一様 [0.02,0.98](official/Helios/M0)/ rescale_t
ワープ格子への整合 / interval 境界 t=0.6 を跨ぐ層化 / 段ごと別。本教師では実効 score の性格が
t=0.6 で不連続に変わるため、一様サンプリングが教師の動作分布から外れる度合いが Hunyuan 時より
大きい可能性がある(機構レベルの推論、未計測)。

### G3. real score の CFG(我々の最大のエビデンス空白)

**D1. CFG スケール** [T1, **E高**] — 教師推論既定 7.5 / 低め固定(DMD 3-8、DMD2 8(SDXL)、
Helios 3.0、SF 3.0、CausVid 3.5、SenseFlow 4.0 + U(3,10) ランダム化変種、MDT は VM40/VD100)/
スイープ / 複数スケール蒸留(DMD2 が future work 扱い)。**M0 線は CFG-distilled 教師だったため
この軸を一度も行使していない**。サブ軸 uncond 源: 教師推論は zeros(コード検証済)— それを
踏襲 / 学習済み null / 別構成; z_global と z_proj の uncond は別決定。式規約
(uncond+s(c−u) vs c+s(c−u))もスケール値の意味を変える既知の罠。

**D2. interval / rescale の扱い** [T1, **E高(前例ゼロ)**] — 全 t で常時 CFG(全先行)/
教師推論を忠実再現(t∈[0.6,1.0] のみ 2-forward、guidance_rescale 0.7/0.5 込み)/ 分解して
片方のみ / CFG なし / **教師を先に CFG 蒸留してから DMD2**(FlashVDM Phase-1 や CogView3 の
w-embedding が CFG 焼き込みの前例)。rescale は score 解釈を持たない。常時 CFG は real score を
常に 2-forward にする(コスト2倍)。

**D3. fake score の CFG** [T2, E低] — なし(official は assert で禁止、Helios 0.0)/
real とミラー / 独立スケール。ただし live-CFG 教師での前例はゼロ。

### G4. critic(fake score)

**E1. 形態と初期化** [T2, E低(形態)+ E高(多重度)] — 教師フルコピー(official)/
共有凍結 base + adapter 切替(Helios/M0; adapter 状態の実行時 assert が run-breaker)/
小型代理。**段をまたぐ共有は前例ゼロ**(段ごとに latent 形式が違うため naive 共有は不可)。
MDT-Dist は critic 自体を持たない(凍結教師を再利用、2モデル構成)— sparse 段での
実装コスト削減の対極案として保持。SenseFlow の **IDA**(生成器重みを critic に λ=0.97 で
注入し続ける)は 8-12B 級 flow で「DMD を収束させた」と主張される第3の機構。

**E2. DSM 目的** [T3, E中(空間・t範囲は参照間で実不一致)] — 予測空間: v-target MSE(Helios)/
ε-空間 MSE(official)/ x̂0+元重み付け(DMD 論文の文言)。t 範囲: full 一様(official)vs
2-98%(Helios)。per-t 重み: なし(両参照)vs SNR 系(独立サブ軸)。
生徒サンプルは常に detach(全先行一致)。

**E3. TTUR** [T4, E中低] — 比 1 / 5 / 10 / LR 非対称のみ。観測分布: 5 が最頻
(DMD2-SDXL/ImageNet, SF, CausVid, SwD, RF)だが **official 内でも SD1.5 は 10**、SF の GAN
変種は 1。DMD2 の ablation(彼らの ImageNet 設定)では 1 不安定 / 10 安定だが遅い、
**非同期 LR 単独は更新比 TTUR より劣る**(App C)。意味論: critic 毎 step + 生成器 1/N step
(official; データ消費 N倍)か N 連続 critic か。LR 非対称(Helios: critic 5×低;
LongLive は LoRA 側を逆に 5×高)は更新比と独立だが等価でないレバー。

**E4. critic の条件付けと入力多様体** [T3] — 教師と同一(z_global+z_proj; 前処理共有で
機械的には可能)/ 縮約 / なし。**生徒生成 support を critic の訓練入力に含めるか**は
G7 の support 方針と連動(critic が見たことのない support 上で fake score を聞くことになるか)。

**E5. 訓練レシピ(optimizer 系)** [T4, E中(参照が実際に割れる), 設定級] —
betas: β1=0(Helios の意図的選択 — DMD 勾配ノイズ対策; SF/CF/SF++ も β1=0)vs 既定 0.9
(official/SwD)。wd: 0.01(official)vs 1e-3(Helios)。grad clip: 10.0(両参照コードの事実、
論文は沈黙; critic 別クリップの有無)。lr はアーム別に 4e-8〜1e-4 まで分布(§4 行列参照)。
batch・スケジュールも同様にアーム別。

### G5. GAN 項

**F1. 採否・trunk 結合・段別 tap** [T3, 採否 E中・tap **E高(sparse 前例ゼロ)**, tap/real は構造的] —
採否: なし(CausVid/CF/SF++/LongLive/RF は GAN なしの DMD で成立; MDT も無 GAN だが非 DMD 系;
PyramidalWan-DMD は「不要と判断」)/ あり(official DMD2 では決定的: ImageNet 2.61→1.51;
SwD は MMD と併用; SF は3目的の1つ)。
**trunk 結合**(サブ軸、tap の前提): 共同訓練(DMD2 原典: μ_fake が DSM+GAN 分類の両損失を受ける
= DM 勾配を作る score 推定器自体を GAN が摂動する)/ head-only(trunk は GAN 勾配に凍結)/
完全分離 D。M0 データ点: critic への fusion は bpg1 必須(eager gan_mode は bpg2 で OOM)。
tap: critic 中間特徴 + pooling(segment-mean / attention / max; head の t 条件付けもサブ選択)/
dense SS 段のみ / 独立判別器(SenseFlow は凍結 DINOv2 ViT-L + 多層ヘッド)/ デコード後 occupancy。
M0 データ点(**vecset・CFG 蒸留教師での観測**): GAN なしで inflate→collapse 再現、遅延 GAN で
15k 完走(ただし valid カウンタの偽陽性が判明済みで、効果量は collapse 阻止ほど確かでない)。
**Helios の公開既定は GAN オフ**(gan_version 設定でのみオン+approx-R1 weight 100)という事実も
「GAN は必須」への反証データ点。

**F2. 遅延・スケジュール** [T4, 設定級] — step 0 から(DMD2 SDXL 4-step / SwD)/
固定遅延(M0 start1000; Helios gan_start_step 1000)/ 二相(DMD2 SD1.5: 39k GAN なし→
resume+GAN 2k @lr 5e-7; ImageNet: 400k→resume)/ drift 信号トリガ / 重みランプ。

**F3. real の定義** [T3, **E高**, 構造的] — GT latent(DMD2 は実データ real により教師超えを
**報告**(2D での主張、本設定で成り立つかは未検証); view-aligned エンコードが必要 =
data_toolkit + TRELLIS.2 encoder 構築)/ 教師 rollout(SF は教師生成 70k を real に使った前例;
SwD は訓練データ自体が 100% 教師合成)/ 混合。教師 rollout のコストは実測 flow ~19s/object(§5)。

**F4. 損失型・重み・正則化** [T4, 設定級] — non-saturating softplus(DMD2; 生成器側重み
5e-3 SDXL / 3e-3 ImageNet / 1e-3 SD1.5、**D 側は別重み** 1e-2; Helios は g/d 非対称 5e-2/1e-2)/
hinge(SenseFlow/FlashVDM/PyramidalWan-Adv)/ R3GAN 相対化+R1R2 近似(SF, λ=30)/
approx-R1(Helios, weight 100)。D の t サンプリング: full range で両群同 t 再ノイズ(DMD2 Eq.4)
vs 生徒 anchor 区間 [t_{s-1},t_s] に制限(SF が安定化と報告)。

### G6. 補助損失

**G1. 回帰/ODE 項** [T4, E中] — なし(DMD2 の主張: TTUR+GAN で代替; Table 3 では
回帰 2.62 ≈ TTUR-only 2.61 と**品質等価**で、GAN が追加利得 1.51)/ DMD1 流 λ=0.25
(LPIPS は 3D latent に不在 → 距離計量が開いた問題)/ warmup 限定(B6 の ODE-init に連続)。
保持時のペア生成サブ軸: solver と NFE(DMD1 は Heun 18/256, PNDM 50)、ペア規模(1k〜16k が
動画系の分布)、コストは §5 から計算可(flow ~19s/object)。
PyramidalWan は重み 0.01 の教師回帰を安定化に併用。ode weight 1→80 は M0 backlog の最安レバー。
**G2. VM/VD(MDT 型)** [T4] — critic 不要の代替コア。TRELLIS v1 の SS+SLAT を 25→1-2 step
にした直接前例(lr 4e-8/1e-8、CFG 40/100、d_norm 正規化)。DMD2 との併用 or 段別使い分け。
**G3. MMD(SwD の PDM)** [T4] — fake-DM 中間特徴の mean-token 線形 MMD。単独でも蒸留として
成立(SwD は SDXL/Wan を MMD のみで出荷; 「DMD は教師が低解像度で無能だと劣化する」が理由)。
**G4. 教師 refinement anchoring(SF++ 型)** [T5] — 教師地平外で生徒 rollout を教師に
補正させ再蒸留。我々では「教師が見たことのない support」への対応として類推可能。
**G5. decoded-space 構造損失** [T4, 本設定固有] — SS の occupancy BCE 等。根拠: SS occupancy は
二値構造で、latent-MSE の近さと復号構造の正しさは乖離し得る(機構、未測定)。

### G7. カスケード結合(本研究の新規領域)

**H1. 段の訓練順序** [T1, E高(文献沈黙=実験軸として未開拓)] — 上流先 / 下流先 / 独立並行→
合成 / joint / 独立 warmup→coupled 切替。段が独立モデルなので per-stage は機械的に可能。
**H2. handoff の support 分布** [T2, 映像系では E低(student-forced で一致)だが
**離散 support への移転は未検証類推**, 構造的] — teacher-forced(全事前計算可)/ GT-encoded /
student-forced(detach、z_proj 再計算は grid_sample のみで安価)/ スケジュール混合 /
DAgger 型教師補正。勾配はどの選択でも段境界を越えない(argwhere 非微分)— 選択は純粋に
入力分布の問題。追加選択肢: 摂動増強 support(頑健化訓練 — Helios の履歴破損
corrupt_history ratio~1/3 の support 類比)。対極の前例: SwD は GT 縮小で段入力を作り
exposure 処理自体を不要化、CogView3 は relay(再ノイズ handoff)が「下流を蒸留しやすくした」と
明言 — **Pixal3D の LR→HR handoff も upsample+量子化という決定論的 relay であり、下流段が
handoff ノイズに頑健である可能性**は検証可能な仮説。
**H3. support シフトの検出** [T3] — support IoU vs 教師/GT、active-voxel 数分布、閾値感度、
support 源条件付き下流品質。M0 で集約スカラが構造的失敗を2度隠した(grad_norm 共縮小、
偽 8/8)ことが専用信号の根拠。
**H4. 条件の事前計算方針** [T3, E低(H2 にほぼ従属)] — 全事前計算(固定 support のみ有効)/
z_global 事前計算+z_proj オンライン / 全オンライン。S1 は dense 固定 support なので常に全事前計算可。

### G8. sparse / 工学

**I1. varlen 上の DMD2 力学** [T1, 前例ゼロ, 構造的] — segment 縮約(C2)+ flash-attn varlen /
pad-to-max で M0 コード再用 / token-budget bucketing。N は LR 数千〜HR ≤49152(実測 mean 20k)。
**I2. バッチ構成と DDP** [T3] — 固定 object 数 / token budget / 長さソート / padding。
token 分散が per-GPU メモリと勾配の重みを同時に振る。
**I3. dense/sparse コード分割** [T3] — S1 = M0 直訳(4096 固定 dense)で DMD2 バグと sparse
バグを分離するブリングアップ価値 vs 統一 varlen 抽象。M0 の故障史は規約級バグが初期失敗を
支配することを示す。
**I4. 計算予算の形** [T4] — 1段1ジョブ 4×H100 DDP / 多ノード fanout(M0 で smoke 済)/
coupled 訓練は複数 1.3B スタック常駐。LoRA vs full×FSDP の分岐(B5)と連動。M0 gotcha
(autocast cache_enabled=False、DDP find_unused/mark-twice、ckpt monkeypatch)の varlen 下での
再検証が前提。実測 peak 39GB(推論時、bf16)が出発点。

### G9. データ

**J1. (image, camera, latent) の調達と規模** [T1, 構造的] — 既知カメラで再レンダ
(data_toolkit)/ in-the-wild + MoGe-2 推定 / 教師 rollout のみ(GT latent なし)。規模:
~5k(M0 規模)vs ~500k(TRELLIS500K、前処理コスト大)。**DMD2 コア(DM 損失+critic DSM)は
noise+条件しか消費しない** — GT latent が要るのは GAN-real / 回帰項 / 評価のみ(official DMD2 は
プロンプト 3M + GAN 用実画像 500k という構成だった)。view-aligned ゆえカメラ誤りは GT latent を
誤った座標系に置く。規模の効果は M0 線でも未測定(undertraining vs ceiling は未決のまま)。

### G10. 評価・監視(事実: M0 では観測性の故障が2度起きた — 偽 8/8、logging aliasing)

**K1. 評価プロトコル** [T3, E高(view-aligned sparse カスケードの標準は不在), frame 規約は構造的] —
scale-honest 層(rigid-only)+ 標準指標の上位集合 / 入力視点座標系での GT 比較 vs canonical /
rigid-ICP(TRELLIS 系 Y↔Z 規約)/ 段別診断(support IoU、GT-support 条件付き品質)。
**K2. validity 定義** [T3] — メッシュ抽出成功のみ / extent-collapse 分類器 / CD 閾値 /
contact-sheet 監査 / 複合。M0 の偽 8/8(box-filling scatter を valid 計上)が直接の根拠。
sparse 版の検出器は未定義。
**K3. 多様性・ぼけの検出** [T4] — multi-seed 距離 / 教師比分布指標 / 高周波幾何統計 /
同一 support 上の段別教師比較 / tight-τ F-score。単一 GT 評価系は多様性損失に盲目(構造的盲点)。
ぼけは B6(ODE-init)の病理が出る場所で、粗い τ の Chamfer は弱くしか罰しない。
**K4. pixel-align 保持の測定** [T3, 本教師固有] — 入力視点再投影誤差(シルエット/深度)/
入力カメラ座標系 CD / z_proj 除去 ablation。canonical 指標は構造的に見えない。
**K5. ベースライン床** [T3, K9 の分母定義に依存] — 教師 12-step 天井 + 教師 naive few-step
Euler 床(M0 では品質ゲートの前提条件**とされた** — 理由: 「蒸留が自明な step 削減に勝る」の
反証可能性確保)+ TRELLIS 系蒸留(FlashVDM/MDT)+ TRELLIS.2 教師アーム。
**K6. drift 信号スイート** [T4] — SS: 占有率/active 数 vs GT; shape: per-ch latent std +
復号 extent(short-axis ratio); x0_std; pseudo_grad_norm(必要条件としてのみ)。vecset の
検出器の sparse 類似物は未定義。
**K7. 収束監視の束と頻度** [T4, E低(§1 の運用規則の具体化)] — 固定 probe decode(500 step 級)+
loss 曲線 + critic 健全性 + **単段訓練中も合成カスケード probe を里程標で回すか**(段別 probe が
緑でも合成が劣化し得る)。判定基準自体は §1 の運用規則(設計軸ではない)。
**K8. ckpt 選択と停止** [T4] — last / 周期 eval argmin / drift ゲート早停 / 固定予算+事後
多 ckpt 評価。min-max 非単調(M0: best ≈ 6000/15000; DMD2 自身も best-ckpt resume)。
**K9. レイテンシ会計** [T4, 設定級だが分母の選択が A1 の損得を変える] — forward 数
(CFG 込み)× flow 限定(FlashVDM 流)/ step 数 / end-to-end。実測: step≠forward
(HR 12 step = 21 forward)、HR は token 数で 1〜33s に振れる(§5)。

### G11. 研究ポジショニング [T5]

候補: few-step pixel-aligned 画像→3D(能力)/ 離散 support 結合・異種多段カスケードへの
初 DMD2(手法)/ teacher-forced vs student-forced on 離散 support(分析)/ 教師選択
(PyramidalWan の split の一般化検証)/ 正直なレイテンシ会計の系統的結果。
各候補は必須 ablation 行列(両 handoff アーム、両教師、…)を変える。

## 3. 軸の依存構造(コード化前に決める順)

T0: A1 stage-scope, A2 teacher-choice, A3 convention-gates
→ T1: D1/D2 CFG, J1 データ, H1 順序, I1 sparse 力学, B1-B3+B5 生成器基本
→ T2: H2 handoff, B4/B6 init, E1 critic 多重度, C1/C2/C4
→ T3: H4 precompute, F1/F3 GAN 構造, K1-K6 評価・信号(K6 drift 信号は K2/H3 の前提), E2/E4, C3, I2/I3
→ T4: G1-G5 補助損失, E3/E5, F2/F4, B7, I4, K7-K9(ただし K9 の分母定義は K5 より先)
→ T5: G11

構造的(覆すと高い): A1, A2(warm-start), A3, J1, H2, I1, E1, F1(trunk/tap)/F3(plumbing),
K1(frame 規約), I4(部分)。設定級(後から振れる): CFG スケール値(**ODE-init 採用時を除く** —
ペアが CFG 構成を埋め込むため), H1(独立訓練なら), B1/B2(1-step 化を除く), 補助損失の重み,
F2, E3, E5, K7-K9。

## 4. 検証済み参照レシピ行列(2026-06-10、全行 arXiv/コード引用で検証済み)

Helios も1行。各セルは実装事実(推奨ではない)。「unv.」= 当該論文・コードから検証不能。

| 手法 (arXiv) | 領域 | steps | init | real-CFG | critic / TTUR | GAN | 補助損失 | EMA | カスケード/rollout |
|---|---|---|---|---|---|---|---|---|---|
| DMD (2311.18828) | 画像 | 1 | 教師コピー | あり 3/8 固定 | 教師フルコピー, 比1(明記なし・交互更新から実質) | なし | **回帰 LPIPS λ0.25(安定化の本体)** | なし | なし(再ノイズのみ) |
| DMD2 (2405.14867) | 画像 | 1, 4 (999/749/499/249) | 教師(1-step は ODE 回帰前置 10k pairs) | あり 8(SDXL)/1.75(SD1.5) | フル第2コピー, **比5**(SDXL/ImageNet)**/10**(SD1.5), 同一 lr, clip10 | **あり**: fake-UNet bottleneck conv ヘッド, real=実画像 500k, 重み 5e-3/3e-3/1e-3; SD1.5/ImageNet は二相, SDXL4step は最初から | 回帰撤廃 | **なし** | backward simulation(rollout 入力を生徒自身が生成、ランダム exit) |
| Helios (2603.04379) | **動画** AR chunk | 4-entry list, shift5.0 (headline 3) | **ODE 回帰 ckpt から**, LoRA r256 | あり 3.0 | 共有 base+adapter 切替, 比5 **+ lr 5×差**, β1=0 | **既定オフ**; オン版: critic 層[5,15,25,35]+final の多層 Conv3D ヘッド, real=実動画, start1000, approx-R1 w100 | ODE 段が前置 | **あり** 0.99@750(検証は live: use_ema_validation=false) | **student-forced**(teacher_forcing=false)+ GT 履歴(ratio1.0)+履歴破損 |
| SenseFlow (2506.00523) | 画像 8-12B flow | 4 anchors | 教師コピー | あり 4.0(+U(3,10) 変種) | フルコピー, 比5, **+IDA: 生成器→critic へ重み注入 λ0.97(収束の鍵)** | あり: **凍結 DINOv2 判別器**(critic と分離), hinge, w0.1 | **ISG**(セグメント内教師誘導 MSE) | なし(IDA が逆向き EMA) | なし |
| FlashVDM (2503.16302) | 3D vecset | 5 NFE | 3相連鎖(CFG蒸留→CD→adv) | **CFG を生徒に焼き込み**(w∼U[2,8] 埋め込み) | **critic なし**(CD) | あり(最終 5k step のみ): latent 空間 DiT 特徴ヘッド, real=実3D, hinge λ0.1 | consistency が本体 | あり(CD target 0.999) | なし |
| MDT-Dist (2509.04406) | **3D TRELLIS 2段** | 1-2 /段 | 教師コピー | あり **VM40 / VD100** | **critic なし**(凍結教師再利用、2モデル) | なし | **VM+VD のみ** | なし | **段ごと独立蒸留**、結合は教師方式のまま(離散 support handoff 無処理の前例) |
| CausVid (2412.07772) | 動画 AR | 4 (999/748/502/247) | 教師+**ODE 回帰 1000 pairs** | 3.5 | 教師 init, 比5, lr 2e-6 | なし | ODE-init のみ | なし | **teacher-forced 並列**(後に SF が劣化原因と診断) |
| Self Forcing (2506.08009) | 動画 AR | 4, shift5 | CausVid 流 ODE init 16k pairs | あり 3.0(**real score=14B**、生徒より大) | 1.3B init, 比5(DMD/SiD)**/1**(GAN変種), gen2e-6/critic4e-7, β1=0 | 3目的の1つ(R3GAN+R1R2 近似, real=**教師生成 70k**, batch768 要, D の t を anchor 区間 [t_{s-1},t_s] に制限=安定化と報告) | DMD/SiD/GAN の択一 | あり 0.99 | **student-forced rollout + KV cache**、最終 denoise step のみ grad |
| Causal Forcing (2602.02214) | 動画 AR | 4 (1/.9375/.8333/.625) | **教師の因子分解を先に整合**→causal ODE 蒸留→DMD | unv.(SF 継承なら 3.0) | SF 継承 | なし | 段階回帰 | unv. | injectivity 定理: 分解不整合 ODE-init→ぼけ解 |
| Self-Forcing++ (2510.02283) | 動画長尺 | 4 | 16k ODE pairs | unv.(teacher/real=**1.3B、生徒と同サイズ** — SF の 14B と対照) | gen2e-6/critic4e-7, 比5, batch8 | なし | **backward noise init** + 窓内 DMD + 任意 GRPO | あり@200ep | 教師地平外を教師補正で再蒸留(DAgger 的) |
| LongLive (2509.22622) | 動画長尺対話 | few(unv.) | SF 流 ODE→DMD→**LoRA streaming 長尺化** | unv. | actor1e-5/critic2e-6(**逆向き lr 差**=LoRA 側) | なし | frame sink / KV recache(構造側) | あり 0.99@200 | **per-clip 教師監督 × student rollout**、過去 detach |
| Rolling Forcing (2509.25161) | 動画長尺 | 5 (1000..200) | 16k ODE pairs | **記載なし** | gen1.5e-6/fake4e-7, 比5 | なし | **SF 目的と 50/50 混合**(正則化) | なし | **window長=step数で軸交絡**、窓内双方向 attention |
| PyramidalWan (2601.04792) | 動画解像度3段 | 2-2-1 (5 NFE) | ピラミッド化 ft 教師から(PT は LoRA) | 記載なし(unv.); 代わりに **w_dmd=教師の競合度で重み付け**(低解像度で無能な元教師対策) | フル流ft, 比/lr unv. | DMD 変種はなし(「不要」); 別系 Adv-OD/PD: 凍結特徴+hinge, rec 項 w2 | 教師回帰 w0.01 | unv. | **教師選択を実験軸化**: 元教師(OT)=定量最良 vs ピラミッド教師(PT*)=視覚最良; 段結合=x̂0 upsample+再ノイズ |
| SwD (2503.16397) | 画像/動画 scale | 4-6(step=scale) | 教師+LoRA r64-128 | あり 4.5-7.5 | 別 LoRA の fake DM, 比5, lr5e-6 | SD3.5/FLUX のみ(SDXL/Wan は **MMD 単独**; 「DMD は低解像度で教師が無能だと劣化」) | **MMD(PDM)w1.0、単独でも成立** | なし | 段入力= **GT(教師合成)縮小**で構成 → exposure 処理自体を不要化; x̂0 を upsample(noisy latent は不可、ablation 済) |
| CogView3-distill (2403.05121) | 画像 relay 2段 | **4+1**(SR は1step) | PD 慣行(unv.) | **w 埋め込みで焼き込み**(round1) | なし(PD) | なし | なし | unv. | **relay(再ノイズ handoff)が下流蒸留を容易化**と明言; 段ごと独立蒸留 |
| Stable Cascade/Würstchen | 画像3段 | — | — | — | — | — | — | — | **few-step 蒸留の公刊なし**(2 検索パスで確認) |
| Pyramid Flow (2410.05954) | 動画 pyramid | — | — | — | — | — | — | — | **蒸留追随なし**; 近傍 GPD(2602.01814)は flat 教師で対象外 |

行列から読める構造(事実の要約、推奨ではない):
- **TTUR は比5 が最頻**だが普遍ではない: official 内でも SD1.5 は 10、SF の GAN 変種は 1、
  DMD1 は実質 1:1。LR 非対称は Helios 系(critic 低)と LongLive(actor 高=LoRA)で逆向き。
- **GAN は「DMD2 系の必須要素」ではない**: CausVid/CF/SF++/LongLive/RF/PyramidalWan-DMD は
  GAN なしの DMD で成立(MDT も無 GAN だが非 DMD 系)、Helios の公開既定もオフ。一方
  official DMD2 では決定的(2.61→1.51)。GAN の代替安定化として IDA(SenseFlow)、MMD(SwD)、
  SF 混合正則化(RF)、教師回帰 w0.01(PyramidalWan)、backward-noise-init(SF++)が分布している。
- **3D の検証行は全て critic-free**(FlashVDM=CD、MDT=VM/VD)— **DMD2 型 critic 訓練の
  3D 公刊前例はゼロ**(E1/G11 に直接効く構造事実)。
- **ODE-init は動画 AR 系で事実上の標準**(1000〜16k pairs)。1-step では official も必須化。
- **real-CFG は全員バラバラ**(記載なし〜100)。教師の推論既定と訓練 CFG の一致を明示した例は
  SwD 程度(行列は各 work の訓練時 CFG のみ記録 — 教師既定との対応は大半で未集計)。
  PyramidalWan の w_dmd(教師競合度重み)は「教師が信頼できる t/解像度だけ強く効かせる」
  という interval-gating の連続版とも読める。
- **段結合**: 独立蒸留+無処理(MDT, CogView3)/ GT 縮小で回避(SwD)/ student-forced
  (SF 系)/ 軸交絡(RF)— と全選択肢に前例がある。離散 sparse support だけが空白。

## 5. 教師プロファイル実測(2026-06-10, job 7928565, n=64+2warmup, H100×1, 1536_cascade)

計測 = `run()` の文単位 inline 鏡像 + ブロック別 CUDA-sync タイマ + flow DiT 全 forward の
CUDA event 計測(branch `exp/profile-teacher`)。失敗 0/66。**v2 の「固定費 ~25s」推定は誤り**
(eval アームの wall から逆算した artifact)。

chunk 集計(mean total **21.86s**/object):

| chunk | mean s | share % |
|---|---|---|
| pre(load+preprocess) | 0.017 | 0.1 |
| camera(MoGe-2) | 0.104 | 0.5 |
| cond(DINOv3+NAF ×4段) | 1.156 | 5.3 |
| support(SS decode+upsample+量子化) | 0.069 | 0.3 |
| **flow(4段サンプリング)** | **19.282** | **88.2** |
| decode(shape+tex+fill_holes) | 0.676 | 3.1 |
| 未計測残差 | 0.557 | 2.5 |

flow 段別(per-forward CUDA event):

| 段 | forwards | CFG steps | ms/forward | 合計 |
|---|---|---|---|---|
| SS | 22 | 10 | 50.0 | 1.10s |
| LR | 21 | 9 | 53.3 | 1.12s |
| HR | 21 | 9 | **513.5** | **10.78s**(min 1.06 / max 32.6s — token 依存) |
| tex | 12 | 0 | 514.3 | 6.17s |

token: N_lr mean 2058 / N_hr mean 20347(max 45879 < cap 49152; 61/64 が 1536 解像度)。
peak GPU 39.2GB(推論、bf16)。注: 段別表は CUDA event の forward 合計で、ブロックタイマとは
LR で ~40ms 差(スケジューラ等のホスト側オーバーヘッド)。

含意(K9 の素材): geometry のみのパイプは end-to-end ~14.8s(= 21.86 − tex 系 7.05s)、
うち flow 13.1s / 非 flow ~1.7s(+未計測残差 ~0.6s)→ **flow が ~88%**。
各段 1-step 化の理論下限 ≈ flow ~0.62s(SS 50ms + LR 53ms + HR 513ms)で、
end-to-end ~14.8s → ~2.3s(**約 6.3×**; 残差を保守的に含めても ~5.7× 以上)。
HR の token 依存分散(1〜33s)は蒸留後も per-forward に残る。
注: 本計測はメッシュ obj 書き出し・GLB ベイクを含まない(eval アームの ~42s/object との差は
export/IO・ハーネス起因と**推定**(未計測))。

## 6. ステータス

- 全論点未決。判断材料: §4 行列(完了)、§5 実測(完了)、A2 の重み比較(未)、
  M0 線の RUN1/RUN2(ODE-init 比較、実行中)。
- 予備実験は main に入れない(branch `exp/profile-teacher` 参照)。
