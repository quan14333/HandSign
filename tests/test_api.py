"""HTTP contract tests; inference is mocked to avoid model downloads."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import api

class PracticeApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.label = api.lesson_labels()[0]

    def test_labels_and_sample(self):
        response = self.client.get('/lessons')
        self.assertEqual(response.status_code, 200)
        self.assertIn(self.label, response.json())
        sample = self.client.get('/samples/' + self.label)
        self.assertEqual(sample.status_code, 200)
        self.assertTrue(sample.headers['content-type'].startswith('video/'))
        self.assertEqual(self.client.get('/samples/unknown-label').status_code, 404)

    def test_model_result_and_temporary_cleanup(self):
        paths = []
        output = {'status': 'correct', 'form': {'score': 91, 'status': 'excellent', 'region_scores': {'right_hand': 95, 'face': 75}}, 'feedback': [{'message': 'Khớp tốt'}]}
        def evaluate(label, path, directory):
            self.assertEqual(label, self.label)
            self.assertEqual(path.read_bytes(), b'recording')
            paths.append(directory)
            return output
        with patch.object(api, 'evaluate_video', side_effect=evaluate):
            response = self.client.post('/evaluate', data={'label': self.label}, files={'video': ('clip.webm', b'recording', 'video/webm')})
        self.assertEqual(response.json(), output)
        self.assertFalse(paths[0].exists())

    def test_invalid_empty_busy_and_failure(self):
        def post(label=None, content=b'clip'):
            return self.client.post('/evaluate', data={'label': label or self.label}, files={'video': ('clip.webm', content, 'video/webm')})
        self.assertEqual(post('unknown-label').status_code, 422)
        self.assertEqual(post(content=b'').status_code, 422)
        with patch.object(api, 'evaluate_video', side_effect=RuntimeError('private details')):
            response = post()
            self.assertEqual(response.status_code, 503)
            self.assertNotIn('private details', response.text)
        self.assertFalse(api._evaluation_lock.locked())
        api._evaluation_lock.acquire()
        try:
            self.assertEqual(post().status_code, 503)
        finally:
            api._evaluation_lock.release()
        with patch.object(api, 'MAX_BYTES', 2):
            self.assertEqual(post().status_code, 413)

if __name__ == '__main__':
    unittest.main()
