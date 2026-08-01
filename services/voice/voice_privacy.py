import os
import uuid
import tempfile
import logging

logger = logging.getLogger(__name__)

class VoicePrivacy:
    @staticmethod
    def get_audio_retention_policy() -> bool:
        return os.getenv("VOICE_AUDIO_RETENTION", "false").lower() == "true"

    @staticmethod
    def create_temp_audio_file(content: bytes, suffix: str = ".webm") -> str:
        """
        Creates a randomly named temporary file for audio processing.
        Never placed in project directories.
        """
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
        return temp_path

    @staticmethod
    def cleanup_temp_file(file_path: str):
        """
        Securely deletes the temporary audio file.
        """
        if not file_path:
            return
            
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Securely deleted temporary audio file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete temporary audio file {file_path}: {e}")

    @staticmethod
    def get_audio_duration(content: bytes) -> float:
        """
        Uses mutagen to safely inspect audio duration from in-memory bytes.
        Returns the length in seconds, or None if it cannot be determined.
        """
        import io
        import mutagen
        try:
            audio = mutagen.File(io.BytesIO(content))
            if audio is not None and audio.info is not None:
                return audio.info.length
        except Exception as e:
            logger.error(f"Failed to determine audio duration: {e}")
        return None
