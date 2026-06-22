# 多摄像头眼动仪优化报告

日期：2026-06-15  
项目目录：`F:\我的资料-李炜\刺激\眼动仪\三摄`

## 1. 摘要

本轮优化的目标是比较一摄与四摄方案，并继续提升四摄精度。实验结果显示：

- 最新最佳方案为 **无眼镜 + 四摄 + adaptive pupil + cubic + 两轮标定**。
- 最新四摄结果已经优于历史最好一摄结果。
- 四摄相对一摄的平均误差提升约 **19.9%**，RMSE 提升约 **13.1%**。
- 当前检测率已经达到 **100%**。
- 但按视觉角度估算，当前精度仍未达到成熟消费级或研究级眼动仪标准。

当前最佳标定文件：

```text
calibration_no_glasses_cubic_round2.json
```

当前最佳验证文件：

```text
validation_all_20260615_162421.json
```

## 2. 当前最佳运行参数

```powershell
python "2026-05-13/usb/dual_usb_eye_tracker.py" `
  --backend haar `
  --use-cameras all `
  --camera-indexes 1,2,0,3 `
  --partial-fusion `
  --min-active-cameras 2 `
  --adaptive-pupil `
  --load-calibration calibration_no_glasses_cubic_round2.json `
  --windowed
```

对应重新标定命令：

```powershell
python "2026-05-13/usb/dual_usb_eye_tracker.py" `
  --backend haar `
  --use-cameras all `
  --camera-indexes 1,2,0,3 `
  --partial-fusion `
  --min-active-cameras 2 `
  --adaptive-pupil `
  --adaptive-ridge `
  --robust-fit `
  --cubic `
  --calibration-rounds 2 `
  --samples-per-point 60 `
  --calibration calibration_no_glasses_cubic_round2.json `
  --validate `
  --windowed
```

## 3. 一摄与四摄对比

| 指标 | 一摄最好历史结果 | 最新四摄最佳结果 | 四摄变化 |
|---|---:|---:|---:|
| 标定文件 | `calibration_left_trial2.json` | `calibration_no_glasses_cubic_round2.json` | - |
| 验证文件 | `validation_left_20260611_154501.json` | `validation_all_20260615_162421.json` | - |
| RMSE | 207.9 px | 180.7 px | 好 27.2 px，约 13.1% |
| 平均误差 | 178.3 px | 142.9 px | 好 35.4 px，约 19.9% |
| P95 误差 | 340.4 px | 342.5 px | 差 2.1 px，约 0.6% |
| 检测率 | 100.0% | 100.0% | 持平 |

结论：  
最新四摄方案在平均精度上已经超过一摄，稳定性基本持平。P95 误差几乎相同，一摄略好，但差距很小。

## 4. 四摄优化过程

| 方案 | 标定文件 | RMSE | 平均误差 | P95 | 检测率 |
|---|---|---:|---:|---:|---:|
| 戴眼镜/普通二次优化 | `calibration_adaptive_pupil.json` | 189.3 px | 153.3 px | 356.5 px | 97.8% |
| 无眼镜/单轮二次 | `calibration_no_glasses.json` | 334.1 px | 315.6 px | 493.2 px | 100.0% |
| 无眼镜/两轮二次 | `calibration_no_glasses_round2.json` | 314.4 px | 246.6 px | 730.4 px | 100.0% |
| 无眼镜/两轮 cubic | `calibration_no_glasses_cubic_round2.json` | 180.7 px | 142.9 px | 342.5 px | 100.0% |

关键发现：

- `--adaptive-pupil` 是提升最大的参数。
- 不戴眼镜后检测率明显改善，可以稳定达到 100%。
- 单纯不戴眼镜并不会自动提高精度，还需要更合适的标定模型。
- 两轮标定增加了样本量。
- `--cubic` 对无眼镜、多样本条件下的角落误差改善明显。

## 5. 当前最佳四摄逐点表现

验证文件：`validation_all_20260615_162421.json`

| 点位 | 坐标 | 平均误差 | RMSE | P95 | 检测率 |
|---|---:|---:|---:|---:|---:|
| p0 | (480, 324) | 190.4 px | 224.6 px | 312.4 px | 100.0% |
| p1 | (1440, 324) | 187.8 px | 236.8 px | 494.2 px | 100.0% |
| p2 | (960, 540) | 117.1 px | 166.3 px | 370.4 px | 100.0% |
| p3 | (480, 756) | 110.5 px | 127.0 px | 217.8 px | 100.0% |
| p4 | (1440, 756) | 108.8 px | 114.2 px | 151.8 px | 100.0% |

主要残余问题：

- 左上和右上点平均误差仍偏大。
- 右上点 p1 的 P95 误差最高，说明偶发偏差仍集中在上方区域。

## 6. 戴眼镜影响

戴眼镜时主要问题不是简单的固定偏差，而是眼部检测不稳定：

- 镜片反光会遮挡瞳孔暗点。
- 镜框会干扰 Haar 眼睛检测。
- 四个摄像头的反光角度不同，导致部分摄像头检测失败。
- 当手、杯子、低头姿态遮挡脸部时，四摄检测率会急剧下降。

实测中，戴眼镜或遮挡时曾出现四摄验证 `0/250` 有效样本的情况。摘眼镜并保持无遮挡后，当前最佳方案检测率达到 `500/500`。

## 7. 行业标准对比

眼动仪行业通常使用视觉角度评价精度，而不是像素。Tobii 说明 gaze accuracy 和 precision 通常以 visual angle 表示；SR Research EyeLink 1000 Plus 等研究级设备典型平均精度约为 `0.25°-0.5°`，远程模式常见约 `0.5°`。一些测试规范会把平均验证误差 `0.5°`、单点不超过 `1°` 作为研究级参考门槛。

按常见 24 英寸 1920x1080 屏幕、观看距离约 60 cm 粗略估算：

| 当前指标 | 像素误差 | 估算视觉角度 |
|---|---:|---:|
| 平均误差 | 142.9 px | 约 3.8° |
| RMSE | 180.7 px | 约 4.8° |
| P95 | 342.5 px | 约 9.0° |

判断：

- 检测率已经可用。
- 精度仍未达到研究级行业标准。
- 当前更适合粗略交互、区域判断、大致注视方向估计。
- 还不适合高精度注视点研究、精细 AOI 边界判断或阅读级眼动分析。

## 8. 消费级水平对比

消费级眼动仪的精度通常低于研究级设备，但成熟产品仍显著优于当前普通 USB 摄像头原型。公开资料和评测中可参考：

| 设备/水平 | 典型精度 | 说明 |
|---|---:|---|
| Tobii Eye Tracker 5 独立评测 | 约 `0.74°` | 消费级远程眼动仪，常用于游戏和交互 |
| Tobii EyeX 论文评测 | `<0.6°` | 较早消费级/低成本桌面眼动设备 |
| Pupil Labs Neon | `1.3°-1.8°` | 轻量穿戴式眼动方案 |
| 当前四摄原型 | 平均约 `3.8°`，RMSE 约 `4.8°` | 普通 USB 摄像头 + Haar/adaptive pupil |

判断：

- **检测率方面**：当前最佳结果 `100%`，已经接近或达到消费级可用体验。
- **精度方面**：仍未达到成熟消费级产品水平，和 Tobii Eye Tracker 5 等产品相比大约还有 `3-6` 倍差距。
- **抗干扰方面**：戴眼镜、反光、遮挡、头部姿态变化仍会明显影响检测，尚未达到消费级产品的稳定性。
- **使用场景方面**：适合粗略交互、区域级判断和原型验证；不适合精确光标控制、阅读分析或精细 AOI 判断。

更准确的定位是：

> 当前系统已经达到“消费级原型”水平，但还没有达到“消费级产品”水平。

主要差距来自：

1. 没有红外补光和角膜反光点。
2. Haar 眼睛检测对姿态、眼镜和光照敏感。
3. 普通 USB 摄像头没有统一曝光、畸变校正和稳定几何。
4. 缺少商业设备常见的头动补偿、眼球模型和个体化生理建模。

参考资料：

- Tobii: remote systems typically operate around `0.2°-0.5°`; eye tracking accuracy relies on pupil center and corneal reflection detection.  
  https://www.tobii.com/blog/expert-eye-tracking-tips  
  https://www.tobii.com/resource-center/learn-articles/how-do-eye-trackers-work
- SR Research EyeLink 1000 Plus: typical accuracy `0.25°-0.5°`, remote average around `~0.5°`.  
  https://www.sr-research.com/eyelink-1000-plus/  
  https://www.sr-research.com/wp-content/uploads/2018/01/EyeLink-1000-Plus-Brochure.pdf
- Tobii Eye Tracker 5 consumer specification and independent evaluation.  
  https://help.tobii.com/hc/en-us/articles/360012483818-Specifications-for-Eye-Tracker-5  
  https://www.researchgate.net/publication/359339342_Evaluating_Accuracy_of_the_Tobii_Eye_Tracker_5
- Tobii EyeX and Pupil Labs Neon consumer/prosumer references.  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5429393/  
  https://pupil-labs.com/products/neon/specs
- EyeLink validation reference: mean validation accuracy limit `0.5°`, each point not exceeding `1°`.  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6625505/

## 9. 结论

当前最佳配置已经证明：

1. 四摄优于一摄。
2. 摘眼镜显著提升检测稳定性。
3. `adaptive-pupil + cubic + 两轮标定` 是目前最有效的组合。
4. 系统已经达到原型可用水平，但还没有达到成熟消费级或研究级眼动仪标准。

## 10. 后续优化建议

优先级从高到低：

1. 引入红外补光和红外摄像头。  
   商业眼动仪依赖稳定红外照明、瞳孔中心和角膜反光，这是精度提升的关键。

2. 替换 Haar 检测。  
   Haar 对姿态、遮挡、眼镜、光照非常敏感。可以尝试 MediaPipe 或专门的眼部关键点模型。

3. 固定头部和摄像头几何。  
   现在误差仍受头部姿态影响明显，建议固定摄像头位置，并尽量保证头部距离和高度一致。

4. 增加验证点数量。  
   当前验证只有 5 点，建议扩展到 9 点或 13 点，尤其增加四角和边缘点。

5. 增加在线漂移校正。  
   在使用过程中加入中心点或已知 UI 点做微调，可以降低长时间使用的系统性偏差。
