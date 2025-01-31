from typing import List, Optional, Dict
import traceback


class LanguageDetection(object):
    "To detect language of the user message using lingua-py"

    def __init__(
        self,
        languages: Optional[List] = None,
        min_words: Optional[int] = None,
        low_accuracy: Optional[bool] = False,
    ) -> None:
        "Loading detector with requirements, by default loads all the languages with 0.0 min confidence"

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
        "Runs language detection"

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
            exception_type = type(e).__name__
            exception_message = str(e)
            exception_traceback = traceback.extract_tb(e.__traceback__)
            line_number = exception_traceback[-1].lineno

            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Line Number: {line_number}")
            print("Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            return {"error": 500, "reason": str(e)}
