/**
 * Local Provider
 * Batch transcription using the bundled on-device lemond NPU engine
 */

import { BatchProvider } from './batch-provider.js';

export class LocalProvider extends BatchProvider {
    getName() {
        return 'local';
    }

    async transcribeSegment(wavBytes) {
        await this.invoke('transcribe_audio_segment', {
            audioData: Array.from(wavBytes),
            apiKey: this.apiKey,
            apiService: 'local',
            language: this.language,
            textFormatted: this.smartFormat,
            insertionMode: this.insertionMode,
            voiceCommandsEnabled: this.voiceCommandsEnabled
        });
    }
}
