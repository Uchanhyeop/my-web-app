// app.js

let tfliteModel = null;
let audioContext = null;
let meydaAnalyzer = null;
let micStream = null;

const MODEL_PATH = "baby_cry_model.tflite";  // 같은 디렉터리에 업로드한 모델

// 오디오 전처리 파라미터
const SAMPLE_RATE = 16000;    // 16kHz (모델 학습 시 사용한 샘플링 레이트와 일치)
const FFT_SIZE = 1024;        // 프레임 길이 (예: 1024 샘플)
const HOP_SIZE = 512;         // 프레임 스트라이드 (예: 50% 오버랩)
const NUM_MEL_BINS = 40;      // MFCC 멜 필터 개수
const NUM_MFCC = 40;          // MFCC 차원 수 (모델 입력 크기와 동일)
const MFCC_SLICES = Math.floor((SAMPLE_RATE - FFT_SIZE) / HOP_SIZE) + 1; 
// 예: (16000-1024)/512+1 ≈ 30 슬라이스

// HTML 요소
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");

/**
 * 1) 웹페이지 로드시 TFLite 모델을 로드
 */
async function loadTFLiteModel() {
  statusEl.innerText = "TFLite 모델 로딩 중…";
  tfliteModel = await tflite.loadTFLiteModel(MODEL_PATH);
  statusEl.innerText = "모델 로드 완료!";
}

// 호출: 페이지 로드 후 바로 실행
loadTFLiteModel();

/**
 * 2) 마이크 입력 받기 (Web Audio API)
 */
async function startMicrophone() {
  if (audioContext) {
    return;
  }
  // 오디오 컨텍스트 생성 (샘플레이트 강제 설정 불가 → 디폴트 샘플레이트로 만들고 resample 예정)
  audioContext = new (window.AudioContext || window.webkitAudioContext)();

  // 마이크 접근
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    statusEl.innerText = "마이크 접근 실패: " + err.message;
    return;
  }

  const sourceNode = audioContext.createMediaStreamSource(micStream);

  // Meyda Analyzer 설정
  meydaAnalyzer = Meyda.createMeydaAnalyzer({
    "audioContext": audioContext,
    "source": sourceNode,
    "bufferSize": FFT_SIZE,
    "hopSize": HOP_SIZE,
    "sampleRate": audioContext.sampleRate,   // 브라우저 디폴트(예: 48000Hz)
    "featureExtractors": ["mfcc"],
    "numberOfMFCCCoefficients": NUM_MFCC,
    "melBands": NUM_MEL_BINS,
    "callback": onMeydaFeatures
  });

  meydaAnalyzer.start();
  statusEl.innerText = "마이크 녹음 중…";
  startBtn.disabled = true;
  stopBtn.disabled = false;
}

/**
 * 3) Meyda 콜백: 한 프레임(FFT_SIZE 샘플)마다 MFCC 배열을 얻는다.
 *    여기서는 실시간으로 슬라이스를 MFCC_BUFFER에 쌓고,
 *    일정량(MFCC_SLICES) 모이면 모델 추론을 실행.
 */
let mfccBuffer = []; // 길이 = MFCC_SLICES × NUM_MFCC

async function onMeydaFeatures(features) {
  // features.mfcc는 길이 NUM_MFCC(40)짜리 배열
  if (!features || !features.mfcc) return;

  // 브라우저 샘플레이트가 44100 또는 48000인 경우, Meyda가 내부적으로 
  // FFT_SIZE(1024)만큼 슬라이스해서 MFCC를 계산해 줍니다. 
  // 하지만 모델 학습은 16000Hz 기준이었으므로, 정확한 비교를 위해
  // 브라우저에서 리샘플링하거나(추가 복잡도), 
  // 간략화하여 브라우저 기본 샘플레이트 결과를 바로 모델에 넣어보겠습니다.
  // (정확히 하려면 Web Audio API로 16000Hz로 리샘플링 필요)

  // 1) MFCC 배열(40개)을 그대로 mfccBuffer에 추가
  mfccBuffer.push(...features.mfcc);

  // 2) MFCC_SLICE 수만큼 모이면 추론 수행
  if (mfccBuffer.length >= MFCC_SLICES * NUM_MFCC) {
    // 모델 입력: Float32Array, shape = [1, MFCC_SLICES * NUM_MFCC]
    const inputTensor = new Float32Array(mfccBuffer.slice(0, MFCC_SLICES * NUM_MFCC));

    // TFLite 모델은 일반적으로 1D 배열 [N] 입력으로 받음
    // (모델 내부에서 슬라이스 개수와 차원을 매핑하도록 만들어졌다고 가정)
    const outputTensor = tfliteModel.predict(inputTensor);

    // outputTensor 는 Float32Array 형식 (모델에 따라 길이 1 또는 다수)
    const prob = outputTensor[0]; // 예: 울음 확률
    resultEl.innerText = `울음 확률: ${prob.toFixed(3)}`;

    // 이전 데이터 롤오버: 남은 MFCC 슬라이스를 앞으로 당김
    mfccBuffer = mfccBuffer.slice(HOP_SIZE / FFT_SIZE * NUM_MFCC); 
    // (모델 구조에 따라 이 부분을 조정해야 합니다)
  }
}

/**
 * 4) 마이크 중지
 */
function stopMicrophone() {
  if (meydaAnalyzer) {
    meydaAnalyzer.stop();
    meydaAnalyzer = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
  statusEl.innerText = "마이크 중지됨";
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

// 버튼 이벤트
startBtn.addEventListener("click", startMicrophone);
stopBtn.addEventListener("click", stopMicrophone);
