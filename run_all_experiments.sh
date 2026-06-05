#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/evawang/miniconda3/envs/pgt_auv/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"

DATA_6TUPLE="${DATA_6TUPLE:-$DATA_DIR/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz}"
DATA_MULTI="${DATA_MULTI:-$DATA_DIR/lbl_aqualoc_seq6_multi_anchor_train4000_test1000.npz}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-4000}"
TEST_SAMPLES="${TEST_SAMPLES:-1000}"

EPOCHS_GCN="${EPOCHS_GCN:-300}"
EPOCHS_TGNN="${EPOCHS_TGNN:-300}"
EPOCHS_TGNN_TF="${EPOCHS_TGNN_TF:-200}"
EPOCHS_TGNN_TF_ED="${EPOCHS_TGNN_TF_ED:-200}"
EPOCHS_PURE_TF="${EPOCHS_PURE_TF:-150}"
EPOCHS_PURE_TF_PHY="${EPOCHS_PURE_TF_PHY:-150}"

LOG_EVERY="${LOG_EVERY:-10}"
WINDOW_SIZE="${WINDOW_SIZE:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

MODE="${1:-all}"

run_cmd() {
  echo
  echo "============================================================"
  echo "[RUN] $*"
  echo "============================================================"
  "$@"
}

ensure_python() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
  fi
}

prepare_data() {
  mkdir -p "$DATA_DIR"

  if [[ ! -f "$DATA_6TUPLE" ]]; then
    run_cmd "$PYTHON_BIN" "$ROOT_DIR/prepare_lbl_aqualoc_seq6_6tuple.py" \
      --train_samples "$TRAIN_SAMPLES" \
      --test_samples "$TEST_SAMPLES"
  else
    echo "[SKIP] 6-tuple dataset already exists: $DATA_6TUPLE"
  fi

  if [[ ! -f "$DATA_MULTI" ]]; then
    run_cmd "$PYTHON_BIN" "$ROOT_DIR/prepare_lbl_aqualoc_seq6_multi_anchor.py" \
      --train_samples "$TRAIN_SAMPLES" \
      --test_samples "$TEST_SAMPLES"
  else
    echo "[SKIP] Multi-anchor dataset already exists: $DATA_MULTI"
  fi
}

train_models() {
  run_cmd "$PYTHON_BIN" "$ROOT_DIR/train_pgt_auv_gcn6_delaymask.py" \
    --data_npz "$DATA_6TUPLE" \
    --epochs "$EPOCHS_GCN" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --log_every "$LOG_EVERY" \
    --model_out "$DATA_DIR/auv_gcn6_delay_seq6_model.pt" \
    --history_out "$DATA_DIR/auv_gcn6_delay_history.csv" \
    --curve_out "$DATA_DIR/auv_gcn6_delay_loss_curve.png"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/train_pgt_auv_tgnn_seq6.py" \
    --data_npz "$DATA_6TUPLE" \
    --window_size "$WINDOW_SIZE" \
    --epochs "$EPOCHS_TGNN" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --log_every "$LOG_EVERY" \
    --model_out "$DATA_DIR/auv_tgnn_seq6_model.pt" \
    --history_out "$DATA_DIR/auv_tgnn_seq6_history.csv" \
    --curve_out "$DATA_DIR/auv_tgnn_seq6_loss_curve.png"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/train_pgt_auv_tgnn_transformer_seq6.py" \
    --data_npz "$DATA_6TUPLE" \
    --window_size "$WINDOW_SIZE" \
    --epochs "$EPOCHS_TGNN_TF" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --log_every "$LOG_EVERY" \
    --model_out "$DATA_DIR/auv_tgnn_transformer_seq6_model.pt" \
    --history_out "$DATA_DIR/auv_tgnn_transformer_seq6_history.csv" \
    --curve_out "$DATA_DIR/auv_tgnn_transformer_seq6_loss_curve.png"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/train_pgt_auv_tgnn_transformer_ed_seq6.py" \
    --data_npz "$DATA_6TUPLE" \
    --window_size "$WINDOW_SIZE" \
    --epochs "$EPOCHS_TGNN_TF_ED" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --log_every "$LOG_EVERY" \
    --model_out "$DATA_DIR/auv_tgnn_transformer_ed_seq6_model.pt" \
    --history_out "$DATA_DIR/auv_tgnn_transformer_ed_seq6_history.csv" \
    --curve_out "$DATA_DIR/auv_tgnn_transformer_ed_seq6_loss_curve.png"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/train_pure_transformer_seq6.py" \
    --data_npz "$DATA_6TUPLE" \
    --window_size "$WINDOW_SIZE" \
    --epochs "$EPOCHS_PURE_TF" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --log_every "$LOG_EVERY" \
    --model_out "$DATA_DIR/auv_pure_tf_seq6_model.pt" \
    --history_out "$DATA_DIR/auv_pure_tf_seq6_history.csv" \
    --curve_out "$DATA_DIR/auv_pure_tf_seq6_loss_curve.png"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/train_pure_transformer_physics_seq6.py" \
    --data_npz "$DATA_6TUPLE" \
    --window_size "$WINDOW_SIZE" \
    --epochs "$EPOCHS_PURE_TF_PHY" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --log_every "$LOG_EVERY" \
    --model_out "$DATA_DIR/auv_pure_tf_phy_seq6_model.pt" \
    --history_out "$DATA_DIR/auv_pure_tf_phy_seq6_history.csv" \
    --curve_out "$DATA_DIR/auv_pure_tf_phy_seq6_loss_curve.png"
}

run_ekf() {
  run_cmd "$PYTHON_BIN" "$ROOT_DIR/ekf_seq6_joint_baseline.py" \
    --data_npz "$DATA_MULTI" \
    --save_npz "$DATA_DIR/ekf_seq6_joint_result.npz"
}

plot_models() {
  run_cmd "$PYTHON_BIN" "$ROOT_DIR/plot_auv_gcn6_delay_result.py" \
    --data_npz "$DATA_6TUPLE" \
    --model_path "$DATA_DIR/auv_gcn6_delay_seq6_model.pt" \
    --out_dir "$DATA_DIR"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/plot_auv_tgnn_seq6_result.py"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/plot_auv_tgnn_transformer_seq6_result.py" \
    --data_npz "$DATA_6TUPLE" \
    --model_path "$DATA_DIR/auv_tgnn_transformer_seq6_model.pt" \
    --out_dir "$DATA_DIR"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/plot_auv_tgnn_transformer_ed_seq6_result.py" \
    --model_path "$DATA_DIR/auv_tgnn_transformer_ed_seq6_model.pt"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/plot_pure_transformer_seq6_result.py" \
    --data_npz "$DATA_6TUPLE" \
    --model_path "$DATA_DIR/auv_pure_tf_seq6_model.pt" \
    --out_dir "$DATA_DIR"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/plot_pure_transformer_physics_seq6_result.py" \
    --data_npz "$DATA_6TUPLE" \
    --model_path "$DATA_DIR/auv_pure_tf_phy_seq6_model.pt" \
    --out_dir "$DATA_DIR"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/plot_selected_cdf_seq6.py"

  run_cmd "$PYTHON_BIN" "$ROOT_DIR/plot_tgnn_transformer_paper_traj.py"
}

compare_all() {
  run_cmd "$PYTHON_BIN" "$ROOT_DIR/compare_gnn_ekf_seq6.py"
}

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [mode]

Modes:
  all       Prepare data, train all models, run EKF baseline, plot, compare (default)
  prepare   Only prepare datasets
  train     Only train neural models
  ekf       Only run EKF baseline
  plots     Only generate model plots
  compare   Only generate unified comparison plots

Environment overrides:
  PYTHON_BIN, DATA_DIR, DATA_6TUPLE, DATA_MULTI
  EPOCHS_GCN, EPOCHS_TGNN, EPOCHS_TGNN_TF, EPOCHS_TGNN_TF_ED
  EPOCHS_PURE_TF, EPOCHS_PURE_TF_PHY, WINDOW_SIZE, BATCH_SIZE, LR, WEIGHT_DECAY, LOG_EVERY
EOF
}

main() {
  ensure_python

  case "$MODE" in
    all)
      prepare_data
      train_models
      run_ekf
      plot_models
      compare_all
      ;;
    prepare)
      prepare_data
      ;;
    train)
      prepare_data
      train_models
      ;;
    ekf)
      prepare_data
      run_ekf
      ;;
    plots)
      plot_models
      ;;
    compare)
      compare_all
      ;;
    -h|--help|help)
      print_usage
      ;;
    *)
      echo "Unknown mode: $MODE" >&2
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
