from typing import List, Optional, Dict
from orichain import error_explainer

import asyncio

class LanguageDetection(object):
    """
    Synchronous way to detect language of the user message using lingua-py
    """

    def __init__(
        self,
        languages: Optional[List] = None,
        min_words: Optional[int] = None,
        low_accuracy: Optional[bool] = False,
    ) -> None:
        """Loading detector with requirements, by default loads all the languages with 0.0 min confidence
        Args:
            languages (Optional[List], optional): List of languages to load. Defaults to None.
            min_words (Optional[int], optional): Minimum words in the user message to detect language. Defaults to None.
            low_accuracy (Optional[bool], optional): To enable low accuracy mode. Defaults to False.
        """

        from lingua import Language, LanguageDetectorBuilder

        if languages:
            language_objects = [getattr(Language, lang) for lang in languages]
            detector = LanguageDetectorBuilder.from_languages(*language_objects)
        else:
            detector = LanguageDetectorBuilder.from_all_languages()
        if low_accuracy:
            detector = detector.with_low_accuracy_mode()

        self.detector = detector.with_preloaded_language_models().build()

        self.min_words = min_words

    def __call__(
        self,
        user_message: str,
        min_words: Optional[int] = None,
        add_confidence: Optional[bool] = False,
        iso_code_639_3: Optional[bool] = False,
    ) -> Dict:
        """Runs language detection
        Args:
            user_message (str): User message to detect language
            min_words (Optional[int], optional): Minimum words in the user message to detect language. Defaults to None.
            add_confidence (Optional[bool], optional): To add confidence in the result. Defaults to False.
            iso_code_639_3 (Optional[bool], optional): To get iso code 639-3 instead of 639-1. Defaults to False.
        Returns:
            Dict: Result of language detection
        """

        try:
            result = {"user_lang": None}
            min_words = min_words or self.min_words
            if min_words:
                if len(user_message.split()) < min_words:
                    return result

            output = self.detector.compute_language_confidence_values(text=user_message)

            result["user_lang"] = (
                output[0].language.iso_code_639_1.name
                if not iso_code_639_3
                else output[0].language.iso_code_639_1.name
            )

            if add_confidence:
                result["confidence"] = output[0].value

            return result
        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

class AsyncLanguageDetection(object):
    """
    Asynchronous way to detect language of the user message using lingua-py
    """

    def __init__(
        self,
        languages: Optional[List] = None,
        min_words: Optional[int] = None,
        low_accuracy: Optional[bool] = False,
    ) -> None:
        """Loading detector with requirements, by default loads all the languages with 0.0 min confidence
        Args:
            languages (Optional[List], optional): List of languages to load. Defaults to None.
            min_words (Optional[int], optional): Minimum words in the user message to detect language. Defaults to None.
            low_accuracy (Optional[bool], optional): To enable low accuracy mode. Defaults to False.
        """

        from lingua import Language, LanguageDetectorBuilder

        if languages:
            language_objects = [getattr(Language, lang) for lang in languages]
            detector = LanguageDetectorBuilder.from_languages(*language_objects)
        else:
            detector = LanguageDetectorBuilder.from_all_languages()
        if low_accuracy:
            detector = detector.with_low_accuracy_mode()

        self.detector = detector.with_preloaded_language_models().build()

        self.min_words = min_words

    async def __call__(
        self,
        user_message: str,
        min_words: Optional[int] = None,
        add_confidence: Optional[bool] = False,
        iso_code_639_3: Optional[bool] = False,
    ) -> Dict:
        """Runs language detection
        Args:
            user_message (str): User message to detect language
            min_words (Optional[int], optional): Minimum words in the user message to detect language. Defaults to None.
            add_confidence (Optional[bool], optional): To add confidence in the result. Defaults to False.
            iso_code_639_3 (Optional[bool], optional): To get iso code 639-3 instead of 639-1. Defaults to False.
        Returns:
            Dict: Result of language detection
        """

        try:
            result = {"user_lang": None}
            min_words = min_words or self.min_words
            if min_words:
                if len(user_message.split()) < min_words:
                    return result

            output = await asyncio.to_thread(self.detector.compute_language_confidence_values(text=user_message))

            result["user_lang"] = (
                output[0].language.iso_code_639_1.name
                if not iso_code_639_3
                else output[0].language.iso_code_639_1.name
            )

            if add_confidence:
                result["confidence"] = output[0].value

            return result
        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}
