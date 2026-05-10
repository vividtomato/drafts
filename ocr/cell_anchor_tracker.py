from collections import deque
from collections import Counter
import math

class SimpleBarcodeTracker:
    
    def __init__(self, history_size=7, min_stability=0.6, spatial_threshold=150, 
                 position_span_threshold=150, position_stability_penalty=0.7, cooldown_frames=3):
        # history_size: сколько последних кадров помним для каждой ячейки
        # min_stability: минимальная доля появлений ячейки в истории, чтобы считать её реальной
        # spatial_threshold: максимальное расстояние между центрами штрихкодов (пиксели), чтобы считать их одной ячейкой
        # position_span_threshold: максимальный размах центров ячейки (пиксели) без штрафа
        # position_stability_penalty: коэффициент штрафа за превышение размаха (0.7 = штраф 30%)
        # cooldown_frames: сколько кадров нельзя переключаться после смены
        self.history_size = history_size
        self.min_stability = min_stability
        self.spatial_threshold = spatial_threshold
        self.position_span_threshold = position_span_threshold
        self.position_stability_penalty = position_stability_penalty
        self.cooldown_frames = cooldown_frames
        
        # track_history[текст] = очередь из 0 (не было) и 1 (было)
        self.track_history = {}
        # position_history[текст] = очередь из центров ячейки или None
        self.position_history = {}
        
        self.current_anchor = None
        self.current_anchor_center = None
        self.switch_cooldown = 0  # сколько кадров нельзя переключаться после смены
        self.switch_log = []
    
    def _calculate_center(self, bbox):
        # bbox = [x1, y1, x2, y2]
        # центр прямоугольника = среднее арифметическое противоположных углов
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _distance_between_centers(self, center1, center2):
        # евклидово расстояние между двумя точками
        return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    
    def _group_by_spatial_proximity(self, ocr_detections):
        # вход: список штрихкодов от OCR, у каждого есть text, confidence, bbox
        # выход: список физических ячеек (объединённые близкие штрихкоды)
        
        if not ocr_detections:
            return []
        
        # добавляем каждому штрихкоду его центр для удобства
        for det in ocr_detections:
            det["center"] = self._calculate_center(det["bbox"])
        
        used = [False] * len(ocr_detections)
        groups = []
        
        for i, det in enumerate(ocr_detections):
            if used[i]:
                continue
            
            # начинаем новую группу с текущего штрихкода
            group_detections = [det]
            used[i] = True
            group_center = det["center"]
            
            # ищем все штрихкоды, которые находятся рядом с центром группы
            for j, other in enumerate(ocr_detections):
                if used[j]:
                    continue
                
                dist = self._distance_between_centers(group_center, other["center"])
                if dist < self.spatial_threshold:
                    group_detections.append(other)
                    used[j] = True
                    # после добавления пересчитываем центр группы как среднее арифметическое всех центров
                    all_centers = [d["center"] for d in group_detections]
                    group_center = (
                        sum(c[0] for c in all_centers) / len(all_centers),
                        sum(c[1] for c in all_centers) / len(all_centers)
                    )
            
            # определяем текст группы: самый частый среди всех штрихкодов группы
            texts = [d["text"] for d in group_detections]
            most_common_text = Counter(texts).most_common(1)[0][0]
            # уверенность группы: максимальная среди всех (берём лучший OCR)
            max_confidence = max(d["confidence"] for d in group_detections)
            
            groups.append({
                "text": most_common_text,
                "confidence": max_confidence,
                "center": group_center,
                "raw_detections": group_detections
            })
        
        return groups
    
    def _update_history(self, cells):
        # cells: список ячеек, обнаруженных в текущем кадре
        # обновляем историю: для каждой ячейки добавляем 1 (была) или 0 (не была)
        
        current_texts = {cell["text"] for cell in cells}
        
        # ячейки, которые видны сейчас: добавляем 1 и центр
        for cell in cells:
            text = cell["text"]
            if text not in self.track_history:
                self.track_history[text] = deque(maxlen=self.history_size)
                self.position_history[text] = deque(maxlen=self.history_size)
            
            self.track_history[text].append(1)
            self.position_history[text].append(cell["center"])
        
        # ячейки, которые были в истории, но не видны сейчас: добавляем 0 и None
        for text in list(self.track_history.keys()):
            if text not in current_texts:
                self.track_history[text].append(0)
                self.position_history[text].append(None)
    
    def _calculate_stability(self, text):
        # стабильность = доля кадров, где ячейка появлялась, за последние history_size кадров
        # также штрафуем, если центры ячейки сильно прыгают между кадрами
        
        if text not in self.track_history:
            return 0.0
        
        history = list(self.track_history[text])
        if len(history) == 0:
            return 0.0
        
        appearances = sum(history)
        stability = appearances / len(history)
        
        # проверка стабильности позиции: если размах центров больше порога, применяем штраф
        positions = list(self.position_history[text])
        valid_positions = [p for p in positions if p is not None]
        if len(valid_positions) >= 3:
            # вычисляем размах координат (максимум - минимум) по X и Y
            x_coords = [p[0] for p in valid_positions]
            y_coords = [p[1] for p in valid_positions]
            x_span = max(x_coords) - min(x_coords)
            y_span = max(y_coords) - min(y_coords)
            # берём худший (максимальный) размах
            position_span = max(x_span, y_span)
            
            if position_span > self.position_span_threshold:
                # снижаем стабильность на коэффициент (например 0.7 = штраф 30%)
                stability = stability * self.position_stability_penalty
        
        return stability
    
    def update(self, ocr_detections):
        # ВХОД: список словарей от OCR
        # каждый словарь содержит:
        #   "text": строка с адресом (например "A12")
        #   "confidence": число от 0 до 1, насколько OCR уверен
        #   "bbox": список [x1, y1, x2, y2] координаты прямоугольника
        
        # шаг 1: группируем близкие штрихкоды в физические ячейки
        input_cells = self._group_by_spatial_proximity(ocr_detections)
        
        # шаг 2: обновляем историю для каждой ячейки
        self._update_history(input_cells)
        
        # шаг 3: уменьшаем счётчик переключения (если активен)
        if self.switch_cooldown > 0:
            self.switch_cooldown -= 1
        
        # шаг 4: вычисляем стабильность для каждой видимой ячейки
        candidates = []
        for cell in input_cells:
            text = cell["text"]
            stability = self._calculate_stability(text)
            candidates.append({
                "text": text,
                "stability": stability,
                "center": cell["center"],
                "confidence": cell["confidence"]
            })
        
        # шаг 5: если нет кандидатов, возвращаем предыдущий якорь
        if not candidates:
            return {
                "anchor": self.current_anchor,
                "anchor_center": self.current_anchor_center,
                "confidence": 0.0,
                "just_switched": False,
                "all_visible_cells": []
            }
        
        # шаг 6: выбираем ячейку с максимальной стабильностью
        best_candidate = max(candidates, key=lambda x: x["stability"])
        
        # если даже лучшая ячейка недостаточно стабильна, не меняем якорь
        if best_candidate["stability"] < self.min_stability:
            return {
                "anchor": self.current_anchor,
                "anchor_center": self.current_anchor_center,
                "confidence": best_candidate["stability"],
                "just_switched": False,
                "all_visible_cells": [(c["text"], c["stability"]) for c in candidates]
            }
        
        # шаг 7: проверяем, нужно ли переключить якорь
        if self.switch_cooldown == 0 and self.current_anchor != best_candidate["text"]:
            # переключение!
            self.switch_log.append({
                "from": self.current_anchor,
                "to": best_candidate["text"],
                "frame": None  # можно установить номер кадра извне
            })
            self.current_anchor = best_candidate["text"]
            self.current_anchor_center = best_candidate["center"]
            self.switch_cooldown = self.cooldown_frames  # cooldown на указанное количество кадров
            just_switched = True
        else:
            just_switched = False
            # если якоря ещё не было, устанавливаем
            if self.current_anchor is None:
                self.current_anchor = best_candidate["text"]
                self.current_anchor_center = best_candidate["center"]
        
        # ВЫХОД: словарь с полями
        #   "anchor": текущий адрес дрона (строка или None)
        #   "anchor_center": координаты центра якоря ((x, y) или None)
        #   "confidence": уверенность в текущем якоре (число 0..1)
        #   "just_switched": True если только что произошло переключение
        #   "all_visible_cells": список всех видимых ячеек с их стабильностью
        return {
            "anchor": self.current_anchor,
            "anchor_center": self.current_anchor_center,
            "confidence": best_candidate["stability"],
            "just_switched": just_switched,
            "all_visible_cells": [(c["text"], c["stability"]) for c in candidates]
        }
        
# Пример использования
tracker = SimpleBarcodeTracker(
    history_size=7,
    min_stability=0.6,
    spatial_threshold=150,
    position_span_threshold=150,
    position_stability_penalty=0.7,
    cooldown_frames=3
)

ocr_results = [
    {"text": "A12", "confidence": 0.92, "bbox": [100, 200, 120, 220]},
    {"text": "A12", "confidence": 0.87, "bbox": [105, 195, 125, 215]},
    {"text": "B34", "confidence": 0.45, "bbox": [300, 200, 320, 220]}
]

result = tracker.update(ocr_results)
print(result["anchor"])
