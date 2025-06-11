from controller import Robot, Camera, Lidar, Motor
import numpy as np
import cv2
import torch
import torch.nn as nn
import math
import time
import random  # Adicionado para randomizar viradas em casos de loop

TIME_STEP = 64
ROBOT_MAX_SPEED = 5.0
ROBOT_MAX_ROTATION = 2.0
SAFETY_DISTANCE = 0.7  # Aumentado para mais margem de reação
CRITICAL_DISTANCE = 0.3  # Novo: Distância crítica para forçar virada imediata
GRID_SIZE = 20
CELL_SIZE = 0.225
CONTROL_FREQUENCY = 30
VISION_FREQUENCY = 10
STUCK_THRESHOLD = 10  # Aumentado para detectar preso prolongado
MIN_PROGRESS_DISTANCE = 0.1  # Distância mínima para considerar progresso

# Estados e ações
ESTADOS = {
    'distancia_objetivo': ['PERTO', 'MEDIO', 'LONGE', 'NAO_VISIVEL'],
    'direcao_objetivo': ['FRENTE', 'ESQUERDA', 'DIREITA', 'ATRAS'],
    'obstaculo_frente': ['SIM', 'NAO'],
    'obstaculo_esquerda': ['SIM', 'NAO'],
    'obstaculo_direita': ['SIM', 'NAO'],
    'espaco_navegavel': ['AMPLO', 'ESTREITO', 'BLOQUEADO']
}
ACOES = ['AVANCAR', 'VIRAR_ESQUERDA', 'VIRAR_DIREITA', 'PARAR', 'RECUAR']

# Modelo RNA (mesmo do train.py)
class SimpleYOLO(nn.Module):
    def __init__(self):
        super(SimpleYOLO, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (224 // 32)**2, 512),
            nn.ReLU(),
            nn.Linear(512, 7 * 7 * (3 + 1 * 5)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.view(-1, 7, 7, 3 + 5)

class ObjectDetector:
    def __init__(self):
        self.model = SimpleYOLO()
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()

    def detect(self, img):
        # Pré-processar
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
        tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0)

        with torch.no_grad():
            pred = self.model(tensor)[0]

        # Parse (simples NMS: selecionar com conf > 0.5)
        detections = []
        S = 7
        for i in range(S):
            for j in range(S):
                conf = pred[i, j, 4].item()
                if conf > 0.5:
                    cls_probs = pred[i, j, 5:8].cpu().numpy()
                    cls = np.argmax(cls_probs)
                    confidence = cls_probs[cls] * conf
                    dx = pred[i, j, 0].item()
                    dy = pred[i, j, 1].item()
                    dw = pred[i, j, 2].item()
                    dh = pred[i, j, 3].item()
                    cx = (j + dx) / S
                    cy = (i + dy) / S
                    w = dw
                    h = dh
                    detections.append({
                        'class': ['objetivo', 'obstaculo_caixa', 'obstaculo_bola'][cls],
                        'bbox': (cx - w/2, cy - h/2, w, h),  # Normalizado
                        'confidence': confidence,
                    })

        # Remover duplicatas simples (maior conf)
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)[:10]  # Max 10

        # Debug: Print detections
        print("Neural Network Detections:")
        for det in detections:
            print(f" - Class: {det['class']}, BBox: {det['bbox']}, Confidence: {det['confidence']:.2f}")
        if not detections:
            print(" - No detections")

        return detections

class LidarProcessor:
    def __init__(self, lidar):
        self.lidar = lidar
        self.num_points = 512
        self.angle_range = 2 * np.pi

    def get_sectors(self):
        distances = np.array(self.lidar.getRangeImage())
        distances[distances == float('inf')] = 10.0
        sector_size = self.num_points // 8
        min_dists = [np.min(distances[i*sector_size:(i+1)*sector_size]) for i in range(8)]

        # Debug: Print sector min distances
        print("LIDAR Sector Min Distances:", min_dists)

        return min_dists

class OccupancyGrid:
    def __init__(self):
        self.grid = np.full((GRID_SIZE, GRID_SIZE), 'DESCONHECIDO', dtype=object)
        self.visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)  # Novo: Grid para rastrear visitados

    def world_to_grid(self, x, y):
        ix = int((x + 2.25) / CELL_SIZE)  # Arena -2.25 to 2.25
        iy = int((y + 2.25) / CELL_SIZE)
        if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE:
            return ix, iy
        return None

    def update_from_lidar(self, robot_pos, robot_theta, distances):
        for i, dist in enumerate(distances):
            if dist < 10.0:
                angle = robot_theta + (i / 512 * 2 * np.pi - np.pi)
                ox = robot_pos[0] + dist * np.cos(angle)
                oy = robot_pos[1] + dist * np.sin(angle)
                grid_pos = self.world_to_grid(ox, oy)
                if grid_pos:
                    ix, iy = grid_pos
                    self.grid[iy, ix] = 'OCUPADO'

        # Debug: Print grid update summary (optional, could be verbose)
        # print("Grid updated from LIDAR. Current grid shape:", self.grid.shape)

    def update_from_camera(self, robot_pos, robot_theta, detections, fov, width):
        for det in detections:
            bbox = det['bbox']
            cls = det['class']
            conf = det['confidence']
            if conf > 0.5:
                # Estimar distância (aprox, usando tamanho médio)
                real_size = 0.2 if cls == 'objetivo' else 0.5  # Diâmetro approx
                pixel_size = max(bbox[2], bbox[3]) * width
                focal = width / (2 * math.tan(fov / 2))
                dist = (real_size * focal) / pixel_size if pixel_size > 0 else 5.0
                angle = (bbox[0] + bbox[2]/2 - 0.5) * fov  # Centro normalizado -0.5 to 0.5
                ox = robot_pos[0] + dist * np.cos(robot_theta + angle)
                oy = robot_pos[1] + dist * np.sin(robot_theta + angle)
                grid_pos = self.world_to_grid(ox, oy)
                if grid_pos:
                    ix, iy = grid_pos
                    if cls == 'objetivo':
                        self.grid[iy, ix] = 'OBJETIVO'
                    else:
                        self.grid[iy, ix] = 'OCUPADO'

        # Debug: Print grid update from camera
        # print("Grid updated from camera.")

    def mark_visited(self, robot_pos):
        grid_pos = self.world_to_grid(robot_pos[0], robot_pos[1])
        if grid_pos:
            ix, iy = grid_pos
            self.visited[iy, ix] = True

    def get_unvisited_direction(self, robot_pos, robot_theta, sector_mins):
        # Calcula a direção com mais células desconhecidas/não visitadas
        unexplored_scores = [0] * 8  # Um por setor
        for i in range(8):
            if sector_mins[i] > SAFETY_DISTANCE:  # Setor livre
                # Projetar pontos no setor
                sector_angle_min = robot_theta + (i * np.pi / 4 - np.pi)
                sector_angle_max = sector_angle_min + np.pi / 4
                for dist in np.linspace(0.5, min(sector_mins[i], 5.0), 5):  # Pontos amostrados
                    for angle in np.linspace(sector_angle_min, sector_angle_max, 3):
                        ox = robot_pos[0] + dist * np.cos(angle)
                        oy = robot_pos[1] + dist * np.sin(angle)
                        grid_pos = self.world_to_grid(ox, oy)
                        if grid_pos:
                            ix, iy = grid_pos
                            if self.grid[iy, ix] == 'DESCONHECIDO' and not self.visited[iy, ix]:
                                unexplored_scores[i] += 1
        if sum(unexplored_scores) == 0:
            return None  # Nenhum inexplorado, default
        best_sector = np.argmax(unexplored_scores)
        if best_sector in [1,2]:  # Esquerda
            return 'ESQUERDA'
        elif best_sector in [5,6]:  # Direita
            return 'DIREITA'
        else:  # Frente ou outros
            return 'FRENTE'

class MotionController:
    def __init__(self, motors):
        self.motors = motors
        self.current_vel_left = 0
        self.current_vel_right = 0
        self.max_accel = 2.0  # m/s^2

    def execute_action(self, action, dt):
        target_left = 0
        target_right = 0
        if action == 'AVANCAR':
            target_left = ROBOT_MAX_SPEED
            target_right = ROBOT_MAX_SPEED
        elif action == 'VIRAR_ESQUERDA':
            target_left = -ROBOT_MAX_ROTATION
            target_right = ROBOT_MAX_ROTATION
        elif action == 'VIRAR_DIREITA':
            target_left = ROBOT_MAX_ROTATION
            target_right = -ROBOT_MAX_ROTATION
        elif action == 'PARAR':
            target_left = 0
            target_right = 0
        elif action == 'RECUAR':
            target_left = -ROBOT_MAX_SPEED / 2
            target_right = -ROBOT_MAX_SPEED / 2

        # Suavizar
        if action != "PARAR":
            self.current_vel_left = self.smooth(self.current_vel_left, target_left, dt)
            self.current_vel_right = self.smooth(self.current_vel_right, target_right, dt)
        else:
            self.current_vel_left =  target_left
            self.current_vel_right = target_right

        self.motors['fl'].setVelocity(self.current_vel_left)
        self.motors['bl'].setVelocity(self.current_vel_left)
        self.motors['fr'].setVelocity(self.current_vel_right)
        self.motors['br'].setVelocity(self.current_vel_right)

        # Debug: Print executed action and velocities
        print(f"Executing Action: {action}, Left Vel: {self.current_vel_left:.2f}, Right Vel: {self.current_vel_right:.2f}")

    def smooth(self, current, target, dt):
        diff = target - current
        max_change = self.max_accel * dt
        if abs(diff) > max_change:
            return current + np.sign(diff) * max_change
        return target

class NavigationController:
    def __init__(self):
        self.robot = Robot()
        self.camera = self.robot.getDevice('camera')
        self.lidar = self.robot.getDevice('lidar')
        self.camera.enable(TIME_STEP)
        self.lidar.enable(TIME_STEP)
        self.detector = ObjectDetector()
        self.lidar_proc = LidarProcessor(self.lidar)
        self.grid = OccupancyGrid()
        self.motors = {
            'fl': self.robot.getDevice('front left wheel'),
            'fr': self.robot.getDevice('front right wheel'),
            'bl': self.robot.getDevice('back left wheel'),
            'br': self.robot.getDevice('back right wheel')
        }
        for m in self.motors.values():
            m.setPosition(float('inf'))
            m.setVelocity(0)
        self.motion = MotionController(self.motors)

        # Odometria inicial (do mundo)
        self.pos = np.array([-1.811861, 1.831535])
        self.theta = 0.785  # ~45 deg
        self.wheel_base = 0.4  # Approx para Pioneer3at

        self.last_time = self.robot.getTime()
        self.vision_counter = 0
        self.stuck_counter = 0  # Novo: Contador para detectar loops
        self.last_espaco = None  # Novo: Rastrear estado anterior
        self.prefer_left = True  # Novo: Alternar preferência de virada para busca
        self.last_pos = self.pos.copy()  # Novo: Rastrear posição anterior para detectar falta de progresso
        self.goal_reached = False  # Novo: Flag para finalizar execução

    def update_odometry(self, dt):
        vl = self.motion.current_vel_left
        vr = self.motion.current_vel_right
        v = (vl + vr) / 2
        w = (vr - vl) / self.wheel_base
        self.theta += w * dt
        self.theta = self.theta % (2 * np.pi)
        self.pos[0] += v * np.cos(self.theta) * dt
        self.pos[1] += v * np.sin(self.theta) * dt

        # Marcar célula atual como visitada
        self.grid.mark_visited(self.pos)

        # Debug: Print current position and orientation
        print(f"Robot Position: ({self.pos[0]:.2f}, {self.pos[1]:.2f}), Theta: {self.theta:.2f} rad")

    def extract_states(self, detections, sector_mins):
        states = {}

        # Objetivo
        objective_det = [d for d in detections if d['class'] == 'objetivo']
        objective_in_front = False
        if objective_det:
            det = max(objective_det, key=lambda d: d['confidence'])  # Melhor
            bbox = det['bbox']
            # Distância estimada
            pixel_diam = max(bbox[2], bbox[3]) * self.camera.getWidth()
            focal = self.camera.getWidth() / (2 * math.tan(self.camera.getFov() / 2))
            dist = (0.2 * focal) / pixel_diam if pixel_diam > 0 else 10.0  # Diam 0.2
            if dist < 0.5:  # Ajustado para menor distância 'PERTO'
                states['distancia_objetivo'] = 'PERTO'
            elif dist < 2.0:
                states['distancia_objetivo'] = 'MEDIO'
            else:
                states['distancia_objetivo'] = 'LONGE'
            # Direção
            center_x = bbox[0] + bbox[2]/2 - 0.5  # -0.5 to 0.5
            angle = center_x * self.camera.getFov()  # Em radianos, já que getFov() retorna radianos
            if abs(angle) < math.radians(15):
                states['direcao_objetivo'] = 'FRENTE'
                objective_in_front = True  # Flag para override de obstáculo
            elif angle < 0:
                states['direcao_objetivo'] = 'ESQUERDA'
            else:
                states['direcao_objetivo'] = 'DIREITA'
        else:
            states['distancia_objetivo'] = 'NAO_VISIVEL'
            states['direcao_objetivo'] = 'FRENTE'  # Busca default

        # Obstáculos de LIDAR (setores: 0-7, frente ~3-4)
        front_min = min(sector_mins[3:5])
        left_min = min(sector_mins[1:3])
        right_min = min(sector_mins[5:7])
        states['obstaculo_esquerda'] = 'SIM' if left_min < SAFETY_DISTANCE else 'NAO'
        states['obstaculo_direita'] = 'SIM' if right_min < SAFETY_DISTANCE else 'NAO'

        # Override para frente: Se objetivo está à frente (pela câmera), ignore obstáculo LIDAR (provavelmente é a bola)
        if objective_in_front and states['distancia_objetivo'] == 'PERTO':
            states['obstaculo_frente'] = 'NAO'
            print("Override: Objetivo detectado à frente pela câmera - ignorando obstáculo LIDAR frontal.")
        else:
            states['obstaculo_frente'] = 'SIM' if front_min < SAFETY_DISTANCE else 'NAO'

        # Espaço: Focar apenas em setores frontais/laterais (1 a 6), ignorando trás (0 e 7)
        clear_sectors = sum(1 for d in sector_mins[1:7] if d > SAFETY_DISTANCE)  # Apenas 6 setores relevantes
        if clear_sectors > 4:  # >4 de 6 para 'AMPLO'
            states['espaco_navegavel'] = 'AMPLO'
        elif clear_sectors > 2:
            states['espaco_navegavel'] = 'ESTREITO'
        else:
            states['espaco_navegavel'] = 'BLOQUEADO'

        # Debug: Print extracted states
        print("Extracted States:")
        for key, value in states.items():
            print(f" - {key}: {value}")

        return states, front_min  # Novo: Retornar front_min para checagem crítica

    def infer_action(self, states, front_min):
        # Simular CPT com lógica
        probs = {a: 0.0 for a in ACOES}

        # Checagem crítica: Se muito perto de obstáculo frontal (não objetivo), forçar virada imediata
        if front_min < CRITICAL_DISTANCE and states['distancia_objetivo'] != 'PERTO':
            if states['obstaculo_esquerda'] == 'NAO':
                return 'VIRAR_ESQUERDA'
            elif states['obstaculo_direita'] == 'NAO':
                return 'VIRAR_DIREITA'
            else:
                return 'RECUAR'  # Se ambos bloqueados, recuar
            print("Critical distance detected! Forcing immediate turn/recuar.")

        # Detectar se preso: pouca mudança na posição (>STUCK_THRESHOLD steps e distância < MIN_PROGRESS_DISTANCE)
        pos_diff = np.linalg.norm(self.pos - self.last_pos)
        self.last_pos = self.pos.copy()
        if pos_diff < MIN_PROGRESS_DISTANCE:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter > STUCK_THRESHOLD:  # Forçar RECUAR se preso por longo tempo
            probs['RECUAR'] = 0.8
            probs['VIRAR_ESQUERDA'] = 0.1
            probs['VIRAR_DIREITA'] = 0.1
            print("Long stuck detected! Forcing recuar.")
            self.stuck_counter = 0  # Reset
            # Normalizar e retornar
            total = sum(probs.values())
            if total > 0:
                for a in probs:
                    probs[a] /= total
            print("Bayesian Probs (stuck mode):", {a: f"{p:.2f}" for a, p in probs.items()})
            return max(probs, key=probs.get)

        if states['distancia_objetivo'] == 'PERTO':
            probs['PARAR'] = 1.0  # Forçar 100% PARAR quando perto
            # Debug probs early
            print("Bayesian Probs (early):", probs)
            self.goal_reached = True
            return 'PARAR'  # Retornar diretamente 'PARAR'

        unvisited_dir = self.grid.get_unvisited_direction(self.pos, self.theta, self.lidar_proc.get_sectors())  # Novo: Direção inexplorada

        if states['distancia_objetivo'] == 'NAO_VISIVEL':
            # Busca balanceada: favorecer AVANCAR em AMPLO, priorizando direção inexplorada
            probs['AVANCAR'] = 0.6 if states['espaco_navegavel'] == 'AMPLO' else 0.3  # Aumentado para avanço em espaço aberto
            if unvisited_dir == 'ESQUERDA':
                probs['VIRAR_ESQUERDA'] += 0.3
            elif unvisited_dir == 'DIREITA':
                probs['VIRAR_DIREITA'] += 0.3
            elif unvisited_dir == 'FRENTE':
                probs['AVANCAR'] += 0.3
            else:  # Default alternado
                if self.prefer_left:
                    probs['VIRAR_ESQUERDA'] = 0.3
                    probs['VIRAR_DIREITA'] = 0.1
                else:
                    probs['VIRAR_DIREITA'] = 0.3
                    probs['VIRAR_ESQUERDA'] = 0.1
                self.prefer_left = not self.prefer_left
        else:
            if states['direcao_objetivo'] == 'FRENTE':
                probs['AVANCAR'] = 0.6
            elif states['direcao_objetivo'] == 'ESQUERDA':
                probs['VIRAR_ESQUERDA'] = 0.6
            elif states['direcao_objetivo'] == 'DIREITA':
                probs['VIRAR_DIREITA'] = 0.6

        if states['obstaculo_frente'] == 'NAO':
            probs['AVANCAR'] += 0.2
        else:
            probs['AVANCAR'] = 0
            if states['obstaculo_esquerda'] == 'NAO':
                probs['VIRAR_ESQUERDA'] += 0.5  # Aumentado para priorizar lado livre
            if states['obstaculo_direita'] == 'NAO':
                probs['VIRAR_DIREITA'] += 0.5  # Aumentado
            probs['RECUAR'] += 0.1

        if states['espaco_navegavel'] == 'BLOQUEADO':
            probs['RECUAR'] = 0.7  # Reduzido para dar chance a viradas
            probs['PARAR'] = 0.1
            # Priorizar virada para lado mais livre mesmo bloqueado
            if states['obstaculo_esquerda'] == 'NAO':
                probs['VIRAR_ESQUERDA'] += 0.1
            if states['obstaculo_direita'] == 'NAO':
                probs['VIRAR_DIREITA'] += 0.1
        elif states['espaco_navegavel'] == 'ESTREITO':
            probs['AVANCAR'] = 0  # Zero prob de avançar em estreito para evitar bloqueio
            # Priorizar virada proativa para lado livre, considerando inexplorado
            if unvisited_dir == 'ESQUERDA' and states['obstaculo_esquerda'] == 'NAO':
                probs['VIRAR_ESQUERDA'] += 0.8
            elif unvisited_dir == 'DIREITA' and states['obstaculo_direita'] == 'NAO':
                probs['VIRAR_DIREITA'] += 0.8
            else:
                if states['obstaculo_esquerda'] == 'NAO' and states['obstaculo_direita'] == 'SIM':
                    probs['VIRAR_ESQUERDA'] += 0.7
                    probs['VIRAR_DIREITA'] += 0.1
                elif states['obstaculo_direita'] == 'NAO' and states['obstaculo_esquerda'] == 'SIM':
                    probs['VIRAR_DIREITA'] += 0.7
                    probs['VIRAR_ESQUERDA'] += 0.1
                else:  # Ambos livres ou ambos bloqueados
                    probs['VIRAR_ESQUERDA'] += 0.4
                    probs['VIRAR_DIREITA'] += 0.4
            probs['RECUAR'] += 0.2  # Aumentada chance de recuar em estreito para escapar regiões presas

        # Normalizar
        total = sum(probs.values())
        if total > 0:
            for a in probs:
                probs[a] /= total

        # Debug: Print probs and chosen action
        print("Bayesian Probs:")
        for a, p in probs.items():
            print(f" - {a}: {p:.2f}")
        action = max(probs, key=probs.get)
        print(f"Chosen Action: {action}")

        return action

    def run(self):
        step_counter = 0
        while self.robot.step(TIME_STEP) != -1 and not self.goal_reached:
            current_time = self.robot.getTime()
            dt = current_time - self.last_time
            self.last_time = current_time

            self.update_odometry(dt)

            # Processar visão a cada VISION_FREQUENCY
            self.vision_counter += 1
            if self.vision_counter % (CONTROL_FREQUENCY // VISION_FREQUENCY) == 0:
                image_data = np.frombuffer(self.camera.getImage(), np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
                image = cv2.cvtColor(image_data, cv2.COLOR_BGRA2BGR)
                detections = self.detector.detect(image)
            else:
                detections = []  # Manter anterior? Ou vazio

            distances = self.lidar.getRangeImage()
            sector_mins = self.lidar_proc.get_sectors()

            self.grid.update_from_lidar(self.pos, self.theta, distances)
            self.grid.update_from_camera(self.pos, self.theta, detections, self.camera.getFov(), self.camera.getWidth())

            states, front_min = self.extract_states(detections, sector_mins)  # Novo: Receber front_min
            action = self.infer_action(states, front_min)

            self.motion.execute_action(action, dt)

            # Para reduzir spam, print separator a cada step
            print(f"--- Step {step_counter} End ---")
            step_counter += 1

        # Após loop: Forçar parar completamente
        self.motion.execute_action('PARAR', 0)
        print("Simulation ended. Robot stopped.")

controller = NavigationController()
controller.run()
