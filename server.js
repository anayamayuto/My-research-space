const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;
const upload = multer({ dest: 'uploads/' });

// CORSを有効化
app.use(cors());
app.use(express.json());

// OpenAI APIキー
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// ffmpeg setup
const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('ffmpeg-static');
ffmpeg.setFfmpegPath(ffmpegPath);

// 文字起こしAPI
app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
  const audioFile = req.file;
  if (!audioFile) {
    return res.status(400).json({ error: 'オーディオファイルが必要です' });
  }

  // 出力ファイルパス (MP3)
  const outputFilePath = path.join('uploads', `audio-${Date.now()}.mp3`);

  console.log(`Converting ${audioFile.path} to ${outputFilePath}...`);

  // ffmpegで変換
  ffmpeg(audioFile.path)
    .toFormat('mp3')
    .on('error', (err) => {
      console.error('An error occurred: ' + err.message);
      // エラー時は元のファイルも削除
      if (fs.existsSync(audioFile.path)) fs.unlinkSync(audioFile.path);
      res.status(500).json({ error: '音声変換に失敗しました' });
    })
    .on('end', async () => {
      console.log('Processing finished!');

      try {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(outputFilePath), {
          filename: 'input.mp3',
          contentType: 'audio/mpeg',
        });
        formData.append('model', 'whisper-1');
        formData.append('response_format', 'verbose_json');
        // formData.append('language', 'ja'); // ユーザー要望により固定を解除
        formData.append('prompt', 'これは日本語の音声の文字起こしです。フィラー（えー、あー）を除去し、句読点を適切につけて、自然な文章で書き起こしてください。');

        const response = await axios.post(
          'https://api.openai.com/v1/audio/transcriptions',
          formData,
          {
            headers: {
              'Authorization': `Bearer ${OPENAI_API_KEY}`,
              ...formData.getHeaders()
            },
            maxBodyLength: Infinity, // axiosのサイズ制限を解除
            maxContentLength: Infinity
          }
        );

        res.json({
          transcription: response.data.text,
          segments: response.data.segments
        });

      } catch (error) {
        console.error('文字起こしエラー:', error.response ? error.response.data : error.message);
        res.status(500).json({ error: '文字起こし処理に失敗しました' });
      } finally {
        // クリーンアップ: アップロードされたファイルと変換後のファイルを削除
        if (fs.existsSync(audioFile.path)) fs.unlinkSync(audioFile.path);
        if (fs.existsSync(outputFilePath)) fs.unlinkSync(outputFilePath);
      }
    })
    .save(outputFilePath);
});

// 文字起こし内容から図解プロンプトを生成するAPI
app.post('/api/analyze-transcript', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'テキストが必要です' });
    }

    const response = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content: "You are a helpful assistant that creates safe, educational DALL-E 3 prompts. \n\nGoal: Create a prompt for an illustration that explains the input concept using a specific 'Daily Life Analogy' (Everyday occurrence) so that a struggling student can understand it immediately.\n\nRules:\n1. Output ONLY the prompt text.\n2. NO Meta-text.\n3. SAFETY CRITICAL: Do NOT use words like 'execution', 'death', 'kill', 'prison', 'blood', 'weapon'.\n4. STRATEGY: CONVERT ABSTRACT CONCEPTS INTO CONCRETE DAILY LIFE SCENARIOS. (e.g., Network -> Traffic jam on a highway; CPU -> A chef in a kitchen; Data -> Packages being delivered; Virus -> A photocopy machine making mistakes). \n5. TEXT: You MUST include instructions to write key labels in JAPANESE (e.g., 'Label the chef as \"CPU\"', 'Text \"データ\" on the box'). \n6. Style: Friendly illustration, clear everyday scene, bright colors, educational but relatable. \n7. IMPORTANT: The description of the scene must be in English for DALL-E, but specify that the TEXT inside the image should be Japanese."
          },
          {
            role: "user",
            content: `Please create an image prompt that explains this concept using a daily life analogy, with Japanese text labels:\n${text}`
          }
        ],
        max_tokens: 300
      },
      {
        headers: {
          'Authorization': `Bearer ${OPENAI_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    let imagePrompt = response.data.choices[0].message.content;
    // Remove "Prompt:" prefixes if present
    imagePrompt = imagePrompt.replace(/^(Prompt:|Here is the prompt:|Request:)\s*/i, '').replace(/\*\*/g, '').trim();

    res.json({ imagePrompt });
  } catch (error) {
    console.error('分析エラー:', error.response ? error.response.data : error.message);
    res.status(500).json({ error: 'テキスト分析に失敗しました' });
  }
});

// 画像生成API
app.post('/api/generate-image', async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: 'プロンプトが必要です' });
    }

    console.log("Generating image with prompt:", prompt);

    const response = await axios.post(
      'https://api.openai.com/v1/images/generations',
      {
        model: "dall-e-3",
        prompt: prompt,
        n: 1,
        size: '1024x1024',
        quality: "standard",
        response_format: "url"
      },
      {
        headers: {
          'Authorization': `Bearer ${OPENAI_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    res.json({ imageUrl: response.data.data[0].url });
  } catch (error) {
    console.error('画像生成エラー详情:', error.response ? JSON.stringify(error.response.data) : error.message);
    res.status(500).json({ error: '画像生成に失敗しました' });
  }
});

// 画像プロキシAPI (CORS対策)
app.get('/api/proxy-image', async (req, res) => {
  try {
    const { url } = req.query;
    if (!url) {
      return res.status(400).send('URL missing');
    }

    const response = await axios.get(url, { responseType: 'stream' });

    res.set('Content-Type', response.headers['content-type']);
    res.set('Access-Control-Allow-Origin', '*');

    response.data.pipe(res);
  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(500).send('Proxy failed');
  }
});

// ==================== 姿勢データAPI ====================

// pose_dataディレクトリの自動作成
const poseDataDir = path.join(__dirname, 'pose_data');
if (!fs.existsSync(poseDataDir)) {
  fs.mkdirSync(poseDataDir, { recursive: true });
}

// 姿勢データ保存API
app.post('/api/pose-data', (req, res) => {
  try {
    const { sessionId, dataPoints } = req.body;
    if (!sessionId || !dataPoints || !Array.isArray(dataPoints)) {
      return res.status(400).json({ error: 'sessionIdとdataPointsが必要です' });
    }

    const filePath = path.join(poseDataDir, `${sessionId}.json`);

    let sessionData;
    if (fs.existsSync(filePath)) {
      const raw = fs.readFileSync(filePath, 'utf-8');
      sessionData = JSON.parse(raw);
    } else {
      sessionData = {
        sessionId,
        startTime: new Date().toISOString(),
        dataPoints: [],
      };
    }

    sessionData.dataPoints.push(...dataPoints);
    fs.writeFileSync(filePath, JSON.stringify(sessionData, null, 2));

    console.log(`[Pose] Session ${sessionId.slice(0, 8)}: +${dataPoints.length} pts (total: ${sessionData.dataPoints.length})`);
    res.json({ success: true, totalPoints: sessionData.dataPoints.length });
  } catch (error) {
    console.error('姿勢データ保存エラー:', error.message);
    res.status(500).json({ error: '姿勢データの保存に失敗しました' });
  }
});

// 姿勢データ履歴取得API
app.get('/api/pose-history/:sessionId', (req, res) => {
  try {
    const { sessionId } = req.params;
    const filePath = path.join(poseDataDir, `${sessionId}.json`);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'セッションが見つかりません' });
    }

    const raw = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(raw);
    res.json({ data });
  } catch (error) {
    console.error('姿勢データ取得エラー:', error.message);
    res.status(500).json({ error: '姿勢データの取得に失敗しました' });
  }
});

// 姿勢データセッション一覧API
app.get('/api/pose-sessions', (req, res) => {
  try {
    const files = fs.readdirSync(poseDataDir).filter(f => f.endsWith('.json'));
    const sessions = files.map(f => {
      const raw = fs.readFileSync(path.join(poseDataDir, f), 'utf-8');
      const data = JSON.parse(raw);
      return {
        sessionId: data.sessionId,
        startTime: data.startTime,
        dataPointCount: data.dataPoints.length,
      };
    });
    res.json({ sessions });
  } catch (error) {
    console.error('セッション一覧エラー:', error.message);
    res.status(500).json({ error: 'セッション一覧の取得に失敗しました' });
  }
});

// ==================== AI学習用データエクスポートAPI ====================

// JSONL / CSV エクスポート
app.get('/api/export/:sessionId', (req, res) => {
  try {
    const { sessionId } = req.params;
    const format = req.query.format || 'jsonl'; // default: jsonl
    
    // Find file that starts with the sessionId (allows using short ID)
    const files = fs.readdirSync(poseDataDir);
    const targetFile = files.find(f => f.startsWith(sessionId) && f.endsWith('.json'));

    if (!targetFile) {
      return res.status(404).json({ error: 'セッションが見つかりません' });
    }

    const filePath = path.join(poseDataDir, targetFile);
    const raw = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(raw);
    const points = data.dataPoints || [];

    if (format === 'csv') {
      // CSV export
      const headers = [
        'timestamp', 'videoTime', 'elapsedSec',
        'shoulderTilt', 'headForward', 'bodyLean', 'postureOverall',
        'camRotX', 'camRotY', 'camRotZ',
        'headYaw', 'headPitch', 'headRoll',
        'lookingAt', 'gazeStability',
        'isLookingAtScreen', 'postureQuality', 'secSinceLastMove',
        'emotionFrown', 'emotionSmile', 'emotionEyeOpenness',
        'audioVolume',
        'lastInteraction', 'secSinceInteraction',
        'handsVisible', 'isTouchingFace', 'currentContext'
      ];

      let csv = headers.join(',') + '\n';
      for (const p of points) {
        const posture = p.posture || p.scores || {};
        const gaze = p.gaze || {};
        const camRot = gaze.cameraRotation || {};
        const headDir = gaze.headDirection || {};
        const eng = p.engagement || {};
        const emo = p.emotion || {};
        const aud = p.audio || {};
        const int = p.interaction || {};
        const ctx = p.context || {};
        const hnds = p.hands || {};

        // Escape quotes for CSV
        let text = ctx.currentText || '';
        if (text) {
          text = '"' + text.replace(/"/g, '""') + '"';
        }

        const row = [
          p.timestamp || '',
          p.videoTime || 0,
          p.elapsedSec || 0,
          posture.shoulderTilt || 0,
          posture.headForward || 0,
          posture.bodyLean || 0,
          posture.overall || 0,
          camRot.x || 0, camRot.y || 0, camRot.z || 0,
          headDir.yaw || 0, headDir.pitch || 0, headDir.roll || 0,
          gaze.lookingAt || 'other',
          gaze.gazeStability || 0,
          eng.isLookingAtScreen || false,
          eng.postureQuality || 'bad',
          eng.secondsSinceLastMovement || 0,
          emo.frown || 0, emo.smile || 0, emo.eyeOpenness || 0,
          aud.volumeLevel || 0,
          int.lastAction || 'none', int.timeSinceAction || 0,
          hnds.handsVisible || 0, hnds.isTouchingFace || false, text
        ];
        csv += row.join(',') + '\n';
      }

      res.setHeader('Content-Type', 'text/csv; charset=utf-8');
      res.setHeader('Content-Disposition', `attachment; filename="${sessionId}.csv"`);
      res.send(csv);

    } else {
      // JSONL export (1 line per data point)
      let jsonl = '';
      for (const p of points) {
        const posture = p.posture || p.scores || {};
        const gaze = p.gaze || {};
        const camRot = gaze.cameraRotation || {};
        const headDir = gaze.headDirection || {};
        const eng = p.engagement || {};
        const emo = p.emotion || {};
        const aud = p.audio || {};
        const int = p.interaction || {};
        const ctx = p.context || {};
        const hnds = p.hands || {};

        const flat = {
          t: p.elapsedSec || 0,
          vt: p.videoTime || 0,
          ts: p.timestamp || 0,
          shoulder: posture.shoulderTilt || 0,
          head: posture.headForward || 0,
          lean: posture.bodyLean || 0,
          overall: posture.overall || 0,
          camX: camRot.x || 0,
          camY: camRot.y || 0,
          camZ: camRot.z || 0,
          yaw: headDir.yaw || 0,
          pitch: headDir.pitch || 0,
          roll: headDir.roll || 0,
          lookAt: gaze.lookingAt || 'other',
          stable: gaze.gazeStability || 0,
          engaged: eng.isLookingAtScreen || false,
          posQuality: eng.postureQuality || 'bad',
          idleSec: eng.secondsSinceLastMovement || 0,
          frown: emo.frown || 0,
          smile: emo.smile || 0,
          eyeOpen: emo.eyeOpenness || 0,
          vol: aud.volumeLevel || 0,
          action: int.lastAction || 'none',
          actionSec: int.timeSinceAction || 0,
          hands: hnds.handsVisible || 0,
          touchFace: hnds.isTouchingFace || false,
          context: ctx.currentText || ''
        };
        jsonl += JSON.stringify(flat) + '\n';
      }

      res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
      res.setHeader('Content-Disposition', `attachment; filename="${sessionId}.jsonl"`);
      res.send(jsonl);
    }

    console.log(`[Export] ${sessionId.slice(0, 8)} → ${format} (${points.length} rows)`);
  } catch (error) {
    console.error('エクスポートエラー:', error.message);
    res.status(500).json({ error: 'エクスポートに失敗しました' });
  }
});

// ==================== 静的ファイル & ルーティング ====================

// 静的ファイルの提供設定 - カレントディレクトリを指定
app.use(express.static(path.join(__dirname, './')));

// ルートパス（/）へのアクセスに対するハンドラー
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'lecture.html'));
});

// 講義ページ
app.get('/lecture', (req, res) => {
  res.sendFile(path.join(__dirname, 'lecture.html'));
});

// サーバー起動
app.listen(port, () => {
  console.log(`サーバーが起動しました: http://localhost:${port}`);
  console.log(`  - メインページ: http://localhost:${port}/`);
  console.log(`  - 講義空間:     http://localhost:${port}/lecture`);
  console.log(`  - データエクスポート: http://localhost:${port}/api/export/{sessionId}?format=csv`);
});