# 001 — 目標と論点マップ(2026-06-10)

## 1. 目標と決定事項

**Pixal3D を教師とした DMD2 蒸留により、few-step の画像→3D(geometry)生成器を作る。**

決定済み:
- 蒸留手法 = **DMD2**。実装は **Helios の DiT ベース実装に準拠**
  (`3D_dmd`/M0 で Hunyuan3D-2.1 向けに再現・運用済みのコードラインを基礎とする)。
- 教師 = **Pixal3D / TRELLIS.2**(Pixal3D = TRELLIS.2 backbone + ProjectAttention。
  VAE・latent 空間は TRELLIS.2 を凍結流用しており共通)。
- 対象 = geometry(sparse structure + shape)。**texture 段はスコープ外**。
- **計測・報告規約**: 時間等の測定は**できるだけ細かいブロック単位**で行う
  (例: rembg / カメラ推定 / DINOv3 抽出(段別) / NAF / 各 flow 段(CFG 有無の step 別) /
  shape デコード / 後処理)。報告では細粒度に加えて **chunk 単位の集計も併記**する
  (chunk = 条件抽出 / flow サンプリング / デコード / 後処理 など)。

以下は論点の地図であり、解はまだ決めない。

## 2. 主要論点(4軸)

### 2.1 vecset との representation の違い

事実: vecset(Hunyuan)は単段・固定長 `[B,4096,64]`・時間軸のみ。TRELLIS 系は
SS = dense `[B,8,16³]`(=4096 tokens 固定)、shape = 可変長 `SparseTensor[N,32]`
(N ≤ 49152)で、時間軸と直交する構造解像度軸を持つ。

論点:
- per-sample 量(loss / grad-norm / normalizer)の segment 縮約(`coords[:,0]` 基準)
- varlen attention(flash-attn varlen)下での DMD2 実装
- token 数可変に対するバッチ構成・DDP 均衡
- M0 機構を直訳できる範囲(SS 段は固定 4096 dense)と書き直しが要る範囲(SLAT 段)の切り分け

### 2.2 多段カスケードの蒸留戦略

事実: 段間は離散 support(coords)で結合される(occupancy の閾値化 = 非微分)。
HR の support は LR latent に依存(`decoder.upsample` 経由)。

論点:
- どの段を few-step 化するか(全段か一部か)、蒸留の順序
- teacher-forced vs student-coupled — 下流の蒸留に使う support 分布をどう選び、
  exposure bias をどう扱うか
- 段ごとの critic(数・共有の可否)
- NFE 配分(現状 geometry ≈ SS 12×2 + LR 12×2 + HR 12×2 forward、CFG 2-forward 込み)

### 2.3 pixel-align 関連

事実: proj 条件(z_global / z_proj)は DiT 外の前処理で、teacher / student / critic で
共有・事前計算可。カメラ(camera_angle_x / distance / mesh_scale)が必須入力。
教師の x0 分布は view-aligned(canonical ではない)。support 変更時の z_proj 再計算は
grid_sample のみで安価。

論点:
- 蒸留データのカメラ調達: レンダリング既知カメラ vs MoGe-2 推定(in-the-wild)
- view-aligned な x0 分布が DMD2 のどこに波及するか(GAN real の作り方、データ準備、評価)
- pixel-align 性能(= Pixal3D の存在意義)が few-step 化で保持されるかの検証方法

### 2.4 教師の選択 — TRELLIS.2 がベースである事実の活用

事実: Pixal3D は TRELLIS.2 の ft とみられる(`_remap_checkpoint_keys` の存在が状況証拠。
重み比較で確定可能だが未実施)。「TRELLIS.2 を蒸留 → 後から proj 追加」は字面では
不成立(proj 化は条件付け追加だけでなく x0 分布の view-aligned 化を伴い、
few-step 生徒に有効な訓練信号がない)。

論点:
- 最終教師は Pixal3D として、TRELLIS.2 を (a) 先行の機構検証台、(b) 生徒の warm-start 元、
  のどちらか/両方/どちらでもなし、として使うか
- 重み比較(ft 由来か)の実施 — warm-start 戦略と知見移転可能性の判断材料
- TRELLIS 系の既存蒸留前例(MDT-Dist 等)の知見の流用範囲

## 3. あとの論点(4軸の外)

1. **CFG の扱い**: 教師は素の CFG が生きている(strength 7.5 / guidance_interval [0.6,1.0] /
   guidance_rescale。Hunyuan は CFG-distilled だったので不要だった論点)。
   real-score の定義(全 t 固定 vs interval 再現 vs rescale の扱い — rescale は score 解釈不能)、
   real-score が 2-forward になるコスト、CFG 蒸留を前置するか。
2. **時間離散化**: few-step 数(1 vs 4)、anchor 配置(rescale_t=3.0/5.0 のワープ格子上か線形か)、
   1-step を狙うなら ODE-init 問題の再来。
3. **GAN 項の設計**(DMD2 の構成要素): real 分布の選択(GT view-aligned latent vs 教師 rollout)、
   sparse 段での discriminator tap / pooling、遅延 warmup(M0 で必須と判明した知見の移植)。
4. **データ計画**: (image, camera) ペアと latent の規模・調達方法。GT latent を使うなら
   data_toolkit + TRELLIS.2 encoder の前処理パイプライン構築が必要。
5. **評価**: view-aligned 出力の評価規約(入力視点座標系での GT 比較、TRELLIS 系 Y↔Z 規約)、
   段別診断指標(support IoU、段条件付き品質)、ベースライン(教師 12-step、TRELLIS 系蒸留前例)。
6. **学習工学**: 1.3B × 3 コピー + varlen + Elastic checkpointing × DDP の成立性、
   bf16 / flash-attn 制約(M0 の autocast・mark-twice 系 gotcha の再点検)。
7. **収束監視**: 可変長設定での probe decode / pseudo_grad_norm の運用
   (毎 run 必須の監視ルールを sparse 向けに移植)。
8. **研究ポジショニング**: 新規クレームを何に置くか
   (few-step pixel-aligned 生成 / 多段 sparse DiT への DMD2 / cascade 整合性、の比重)。
9. **NFE・レイテンシ会計**: guidance_interval により「step 数」と「実効 forward 数」が乖離する
   (実測: HR は 12 step ≈ 21 forward — 9 step が CFG 2-forward、interval を抜けると tqdm が
   約2倍速になるのをログで確認。tex は strength 1.0 で常時 1-forward なので同トークン数の
   HR の約半分の時間)。CFG の扱い(論点 3-1)はレイテンシに直結する。few-step 化の
   speedup を語るときの分母・分子(step か forward か、CFG 込みか)をどう定義するか。
10. **速度報告プロトコルと固定費**: 教師の実測(H100, 1536_cascade)では end-to-end
   ~42 s/object のうち geometry の flow サンプリングは ~9–13 s(SS ~1.1s / LR ~1.2s /
   HR ~7–11s)で、残りは固定費(条件抽出・デコード・GLB 後処理 ~25s)。few-step 化しても
   固定費は残るため、**speedup 主張は flow 部分で報告し、デコード以降は別立てにする**
   (FlashVDM と同じ流儀)。HR のトークン数(≤49152)依存でレイテンシが 7→11s と振れる
   分散の扱いも明記する。

## 4. 調査: coarse-to-fine / 多段カスケード生成モデルの few-step 蒸留 SOTA

deep-research(101 agents, 19 sources fetch, 95 claims 抽出 → 上位 25 を 3-vote 敵対検証 →
24 confirmed / 1 refuted、2026-06-10 実施)。論点 2.2(多段カスケード戦略)・2.4(教師選択)の判断材料。

### 結論(検証済み文献からの帰納)

「拡散時間軸と直交する階層を持つ生成器の few-step 蒸留」の先行例は
**動画 AR(時間チャンク)カスケード**と**解像度カスケード**に集中し、レシピは4パターンに収斂する:

- **(A) 教師因子分解の事前整合** — 学生の段構造に合わせて教師をまず変換してから ODE-init → DMD。
  Causal Forcing の frame-level injectivity 理論が根拠: 因子分解の異なる教師からの ODE-init は
  conditional-expectation(平均化=ぼけ)解に収束する(Proposition 3.3)。
- **(B) 段間ハンドオフの student-forcing** — 蒸留中の下流段入力は学生自身の上流出力
  (detach して定数文脈化)。teacher-forced 中間状態は劣化原因と診断済み(Self Forcing 系で一貫)。
- **(C) per-stage 教師監督の合成** — 教師は「自分が競合(competent)な段」のみ監督し、
  per-stage 監督の集合で全カスケードを誘導(LongLive)。教師の地平外は教師補正でアンカー(SF++)。
- **(D) カスケード軸の時間軸折り畳み** — 段を別パスにせず step=scale 対応(SwD)や
  重複ノイズ窓(Rolling Forcing)で結合を連続化。

**ただし**: これらの「段」は全て同種(時間チャンク or 同一モデルの解像度)。
**TRELLIS 型の離散 support(active voxel 集合)で結合された異種マルチモデル段カスケードを
直接 few-step 蒸留した検証済み先行研究は確認されなかった = 未踏領域**(我々の新規性の核候補)。

### 検証済み主要文献

| 論文 | 手法 | カスケード軸 | 段結合・exposure bias の扱い | steps |
|---|---|---|---|---|
| CausVid (arXiv:2412.07772, CVPR'25) | DMD | 時間チャンク AR | asymmetric distillation(bidirectional 教師→causal 学生)。文脈は teacher-forced | 50→4 |
| Self Forcing (arXiv:2506.08009, NeurIPS'25 Spotlight) | DMD/SiD/GAN | 時間チャンク AR | **student-forced rollout**(KVキャッシュ付き自己生成文脈)+ 系列全体の holistic 損失 | 4 |
| Causal Forcing (arXiv:2602.02214, ICML'26) | ODE-init + DMD | 時間チャンク AR | **教師をまず学生の因子分解へ変換**(合成データで AR 教師を TF 訓練)→ causal ODE-init → student-forced DMD | few |
| Self-Forcing++ (arXiv:2510.02283) | DMD | 時間(教師地平超え) | 学生の自己生成長尺動画上で short-horizon 教師が refine → 補正を再蒸留(DAgger 的) | 4 |
| LongLive (arXiv:2509.22622, NVIDIA, ICLR'26) | DMD(Self-Forcing 流) | 時間チャンク AR | streaming long tuning: student-forced 延長 + **DMD は新規クリップのみ**・過去フレーム detach。「教師は競合する段のみ監督」 | few |
| Rolling Forcing (arXiv:2509.25161, TencentARC, ICLR'26) | DMD 系 | 時間×ノイズの交絡 | rolling window 内で漸増ノイズを同時デノイズ(window長=step数)。段境界を軟結合化 | 5 |
| PyramidalWan (arXiv:2601.04792, Qualcomm) | DMD/Adv 蒸留比較 | 時空間解像度 3 段 | 結合=決定的ハンドオフ(NN アップサンプル + **再ノイズ境界条件**)。**教師選択を明示比較: 非カスケード元教師(DMD-OT)=定量最良 / カスケード化教師(DMD-PT*)=視覚最良** | 2-2-1 |
| SwD (arXiv:2503.16397, Yandex) | DMD2 統合(MMD+DMD+GAN) | 解像度(学生側に誘導) | 単一学生で step=scale。遷移は x̂0 アップサンプル→再ノイズ。隣接スケール対で joint 訓練 | 4–6 |

REFUTED(1-2): 「PyramidalWan が DMD 中に student-forced backward-simulation で中間段サンプルを
生成する」— 同論文に段間 exposure bias / joint-vs-per-stage の明示的議論は**ない**。
柱4(理論)を直接扱うのは Causal Forcing のみ。

### 我々(Pixal3D/TRELLIS.2 の DMD2)への示唆

1. **(B) は「teacher-forced → student-forced refresh」案の直接の裏付け**。動画系の一致した
   診断(teacher-forced 文脈=劣化原因)は、support の student-forcing を最初から設計に入れる根拠。
2. **(A) の警告が ODE-init に効く**: 1-step 化で ODE-init を使うなら「段分解の整合」が前提。
   我々は教師自体が段分解済み(段ごとに別モデル)なので Causal Forcing の病理(因子分解ミスマッチ)は
   原理的に回避されている — これは**多段教師の意外な利点**として主張できる可能性。
3. **PyramidalWan の教師選択のねじれ**(非カスケード教師=定量最良)は、
   「TRELLIS.2 vs Pixal3D どちらから蒸留するか」(論点 2.4)に直接対応する先行知見。
4. **(D) は将来の対極案**: 段を統合した単一学生(step=stage)への折り畳みは、SS→LR→HR が
   別モデル・別 latent の我々には直接は適用できないが、「LR と HR を単一学生に統合する」
   方向の参照点になる。
5. 段の蒸留**順序**(上流先か下流先か、切替タイミング)を実験軸にした研究は依然未確認 → ここも空白。

### 注意(調査の限界)

- 3D 柱(TRELLIS 系蒸留の Helios/MDT-Dist/FlashVDM **以降**の新作)と画像カスケード古典
  (Stable Cascade 等)は検証を通過した claim が 0 件 — 「存在しない」ではなく「本検証パスで
  未確認」。3D 側は既知 3 本(手元把握済み)以外に新作がない可能性が高いが、断定はしない。
- 動画系の「段」は同種・同一モデル。離散 sparse support への外挿は本質的に類推。
- 分野の動きが速い(言及のみ未検証: Causal Forcing++, BAgger arXiv:2512.12080 等)。

オープンな問い(調査側の出力): 離散 support の handoff に student-forced 蒸留した場合、
support 誤りに対して下流教師スコアは有効な修正勾配を与えるか / injectivity 論は構造軸に類推可能か /
教師選択のねじれは一般現象か / 蒸留順序の curriculum はオープン。

## 5. ステータス

- 全論点とも未決。次は判断材料の収集(2.4 の重み比較が最安・最初の候補)。
- 教師の細粒度プロファイル計測は予備実験として **branch `exp/profile-teacher`** に分離
  (`scripts/profile_teacher.{py,sh}`、Toys4k 66 objects)。予備実験は main に入れない方針。
