# Multi-Camera Eye Tracking Prototype

这是一个基于普通 USB 摄像头的多摄像头眼动仪原型项目。项目主要验证一摄、双摄和四摄条件下的眼动追踪精度，并记录了不同标定与融合参数的实验结果。

## 当前最佳结果

当前最佳方案：

```text
四摄 + 无眼镜 + adaptive pupil + cubic + 两轮标定
```

最佳标定文件：

```text
calibration_no_glasses_cubic_round2.json
```

最佳验证文件：

```text
validation_all_20260615_162421.json
```

核心指标：

| 指标 | 结果 |
|---|---:|
| RMSE | 180.7 px |
| 平均误差 | 142.9 px |
| P95 误差 | 342.5 px |
| 检测率 | 100.0% |

相对历史最好一摄结果，最新四摄平均误差提升约 19.9%，RMSE 提升约 13.1%。

## 运行当前最佳四摄

```powershell
python "2026-05-13/usb/dual_usb_eye_tracker.py" --backend haar --use-cameras all --camera-indexes 1,2,0,3 --partial-fusion --min-active-cameras 2 --adaptive-pupil --load-calibration calibration_no_glasses_cubic_round2.json --windowed
```

## 重新标定当前最佳四摄

```powershell
python "2026-05-13/usb/dual_usb_eye_tracker.py" --backend haar --use-cameras all --camera-indexes 1,2,0,3 --partial-fusion --min-active-cameras 2 --adaptive-pupil --adaptive-ridge --robust-fit --cubic --calibration-rounds 2 --samples-per-point 60 --calibration calibration_no_glasses_cubic_round2.json --validate --windowed
```

## 主要文件

| 文件 | 说明 |
|---|---|
| `2026-05-13/usb/dual_usb_eye_tracker.py` | 主程序 |
| `calibration_no_glasses_cubic_round2.json` | 当前最佳标定 |
| `validation_all_20260615_162421.json` | 当前最佳验证结果 |
| `eye_tracking_comparison_report_20260617.md` | 一摄与四摄对比报告 |
| `eye_tracking_optimization_report_20260615.md` | 完整优化报告 |
| `eye_tracking_web_report_20260615.html` | Web 版报告 |
| `eye_tracking_site_20260615/` | 静态网站发布目录 |

## 当前算法栈

| 模块 | 当前方案 |
|---|---|
| 检测后端 | OpenCV Haar |
| 瞳孔增强 | adaptive pupil |
| 多摄融合 | partial fusion，至少 2 路有效 |
| 标定拟合 | Ridge regression |
| 异常抑制 | Huber robust fit |
| 映射模型 | cubic 三次多项式 |
| 标定策略 | 两轮 9 点标定，每点 60 样本 |

## 消费级水平判断

当前系统检测率已经达到可用水平，但精度仍未达到成熟消费级产品水平。

| 设备/水平 | 典型精度 |
|---|---:|
| Tobii Eye Tracker 5 独立评测 | 约 0.74 度 |
| Pupil Labs Neon | 约 1.3-1.8 度 |
| 当前四摄原型 | 平均约 3.8 度，RMSE 约 4.8 度 |

当前更适合粗略交互、区域判断和原型验证。若要接近成熟消费级产品，需要进一步引入红外补光、黑白全局快门摄像头、角膜反光点、头动补偿和更强眼部模型。

## 红外升级建议

推荐从双摄红外验证开始：

```text
OV9281 黑白全局快门无 IR-cut 摄像头 x 2
850nm 可调红外补光灯 x 1-2
```

验证有效后再扩展到四摄。

## 报告与网页

本项目包含 Markdown 报告和静态 Web 报告。Web 版本可以直接打开：

```text
eye_tracking_web_report_20260615.html
```

静态网站发布目录：

```text
eye_tracking_site_20260615/
```

