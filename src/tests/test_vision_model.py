import pytest
from pathlib import Path
import base64
from unittest.mock import Mock, patch, mock_open
from src.vision_model import VisionProcessor, MagnifierItem, MagnifierPage

# Test data
MOCK_IMAGE_PATH = "test_image.png"
MOCK_IMAGE_BYTES = b"mock image data"
MOCK_BASE64 = base64.b64encode(MOCK_IMAGE_BYTES).decode('utf-8')

@pytest.fixture
def vision_processor():
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test_key',
        'GEMINI_API_KEY': 'test_key'
    }):
        return VisionProcessor()

@pytest.fixture
def mock_image():
    with patch("builtins.open", mock_open(read_data=MOCK_IMAGE_BYTES)):
        yield MOCK_IMAGE_PATH

def test_magnifier_item():
    """Test MagnifierItem model"""
    item = MagnifierItem(
        page_id=1,
        cycle_id=1,
        page_number="i",
        text_after_symbol="test text"
    )
    assert item.page_id == 1
    assert item.cycle_id == 1
    assert item.page_number == "i"
    assert item.text_after_symbol == "test text"

def test_magnifier_page():
    """Test MagnifierPage model"""
    items = [
        MagnifierItem(
            page_id=1,
            cycle_id=1,
            page_number="i",
            text_after_symbol="test text"
        )
    ]
    page = MagnifierPage(magnifier_items=items)
    assert len(page.magnifier_items) == 1
    assert page.magnifier_items[0].text_after_symbol == "test text"

def test_vision_processor_init(vision_processor):
    """Test VisionProcessor initialization"""
    assert vision_processor.openai_client is not None
    assert vision_processor.gemini_client is not None

@pytest.mark.asyncio
async def test_detect_magnifier_true(vision_processor, mock_image):
    """Test detect_magnifier when magnifier is found"""
    mock_response = Mock()
    mock_response.text = "true"
    
    with patch.object(vision_processor.gemini_client, 'generate_content', 
                     return_value=mock_response):
        result = vision_processor.detect_magnifier(mock_image)
        assert result is True

@pytest.mark.asyncio
async def test_detect_magnifier_false(vision_processor, mock_image):
    """Test detect_magnifier when no magnifier is found"""
    mock_response = Mock()
    mock_response.text = "false"
    
    with patch.object(vision_processor.gemini_client, 'generate_content', 
                     return_value=mock_response):
        result = vision_processor.detect_magnifier(mock_image)
        assert result is False

@pytest.mark.asyncio
async def test_detect_magnifier_error(vision_processor, mock_image):
    """Test detect_magnifier error handling"""
    with patch.object(vision_processor.gemini_client, 'generate_content', 
                     side_effect=Exception("API Error")):
        result = vision_processor.detect_magnifier(mock_image)
        assert result is False

def test_extract_text_success(vision_processor, mock_image):
    """Test successful text extraction"""
    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content="{'page_number': '1', 'text': 'test text'}"))
    ]
    
    with patch.object(vision_processor.openai_client.chat.completions, 'create', 
                     return_value=mock_response):
        result = vision_processor.extract_text(mock_image)
        assert result == {'page_number': '1', 'text': 'test text'}

def test_extract_text_error(vision_processor, mock_image):
    """Test text extraction error handling"""
    with patch.object(vision_processor.openai_client.chat.completions, 'create', 
                     side_effect=Exception("API Error")):
        result = vision_processor.extract_text(mock_image)
        assert result == {"page_number": None, "text": None}

def test_extract_text_invalid_response(vision_processor, mock_image):
    """Test handling of invalid response format"""
    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content="invalid json"))
    ]
    
    with patch.object(vision_processor.openai_client.chat.completions, 'create', 
                     return_value=mock_response):
        result = vision_processor.extract_text(mock_image)
        assert result == {"page_number": None, "text": None} 