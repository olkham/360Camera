from main import Equirectangular360

processor = Equirectangular360('')
print('Testing angle normalization:')
print(f'(0, -5, 0) -> {processor.normalize_angles(0, -5, 0)}')
print(f'(0, 5, 0) -> {processor.normalize_angles(0, 5, 0)}')
print(f'(0, -95, 0) -> {processor.normalize_angles(0, -95, 0)}')
print(f'(0, 95, 0) -> {processor.normalize_angles(0, 95, 0)}')
print(f'(360, 0, 0) -> {processor.normalize_angles(360, 0, 0)}')
print(f'(0, 0, 0) -> {processor.normalize_angles(0, 0, 0)}')
