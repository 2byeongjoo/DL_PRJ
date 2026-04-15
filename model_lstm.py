from tensorflow.keras.models import load_model

loaded_model = load_model('best_model.h5')

# 테스트 데이터로 최종 정확도 측정
final_loss, final_acc = loaded_model.evaluate(X_test, y_test)

print(f"\n테스트 정확도: {final_acc * 100:.2f}%")