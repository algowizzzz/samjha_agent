/**
 * Unit tests for document review frontend WebSocket UI interactions.
 * Tests SocketIO event handling, timeline updates, chat functionality, and room management.
 *
 * Run with: node test/test_doc_review_frontend_websocket.js
 * Or in browser: Load test/test_doc_review_frontend_websocket.html
 */

// Mock SocketIO client
class MockSocketIO {
    constructor() {
        this.listeners = {};
        this.emitQueue = [];
        this.connected = false;
    }

    on(event, handler) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(handler);
    }

    emit(event, data) {
        this.emitQueue.push({ event, data });
    }

    trigger(event, data) {
        const handlers = this.listeners[event] || [];
        handlers.forEach((handler) => handler(data));
    }

    connect() {
        this.connected = true;
        this.trigger('connect');
    }

    disconnect() {
        this.connected = false;
        this.trigger('disconnect');
    }

    clear() {
        this.listeners = {};
        this.emitQueue = [];
    }
}

// Mock DOM elements
function createMockDOM() {
    return {
        chatHistory: { innerHTML: '', scrollTop: 0, scrollHeight: 0 },
        chatInput: { value: '', addEventListener: () => {} },
        btnSendChat: { disabled: false, addEventListener: () => {} },
        chatSpinner: { classList: { toggle: () => {} } },
        chatStatusBadge: { textContent: '' },
        chatAutoExecute: { checked: true },
        timelineFeed: { innerHTML: '' },
        btnClearTimeline: { addEventListener: () => {} },
    };
}

// Test framework (simple assertion library)
class TestRunner {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(message || 'Assertion failed');
        }
    }

    assertEqual(actual, expected, message) {
        if (actual !== expected) {
            throw new Error(
                message || `Expected ${expected}, got ${actual}`
            );
        }
    }

    assertContains(container, item, message) {
        if (!container.includes(item)) {
            throw new Error(message || `Expected container to include "${item}"`);
        }
    }

    async run() {
        console.log('\n🧪 Running Frontend WebSocket Tests\n');
        console.log('='.repeat(60));

        for (const { name, fn } of this.tests) {
            try {
                await fn();
                console.log(`✅ ${name}`);
                this.passed++;
            } catch (error) {
                console.error(`❌ ${name}`);
                console.error(`   ${error.message}`);
                this.failed++;
            }
        }

        console.log('\n' + '='.repeat(60));
        console.log(`\nResults: ${this.passed} passed, ${this.failed} failed\n`);
        return this.failed === 0;
    }
}

// Extract testable functions from the cockpit (simplified versions)
function createTestableFunctions() {
    const mockSocket = new MockSocketIO();
    const mockDOM = createMockDOM();
    let timelineEvents = [];
    let chatMessages = [];
    let chatStatus = 'idle';
    let joinedRoomId = null;
    let selectedDocumentId = null;

    return {
        socket: mockSocket,
        dom: mockDOM,
        timelineEvents,
        chatMessages,
        chatStatus,
        joinedRoomId,
        selectedDocumentId,

        // Timeline functions
        addTimelineEvent(eventType, payload) {
            const timestamp = payload.timestamp || new Date().toISOString();
            timelineEvents.unshift({ eventType, payload, timestamp });
            if (timelineEvents.length > 40) {
                timelineEvents.pop();
            }
            // Update reference
            this.timelineEvents = timelineEvents;
            this.renderTimeline();
        },

        renderTimeline() {
            const events = this.timelineEvents || timelineEvents;
            if (!events.length) {
                mockDOM.timelineFeed.innerHTML = '<div class="empty-state">Live workflow events will appear here.</div>';
                return;
            }
            mockDOM.timelineFeed.innerHTML = events
                .map((event) => {
                    const label = this.formatTimelineLabel(event.eventType, event.payload);
                    return `
                        <div class="timeline-event">
                            <time>${this.formatTimestamp(event.timestamp)}</time>
                            <strong>${label.title}</strong>
                            <p class="mb-0 text-muted">${label.detail}</p>
                        </div>
                    `;
                })
                .join('');
        },

        formatTimelineLabel(eventType, payload = {}) {
            switch (eventType) {
                case 'node_started':
                    return {
                        title: `Node started: ${payload.node || payload.label || 'unknown'}`,
                        detail: payload.summary || '',
                    };
                case 'node_completed':
                    return {
                        title: `Node completed: ${payload.node || payload.label || 'unknown'} (${payload.status || 'success'})`,
                        detail: payload.summary || '',
                    };
                case 'agent_plan_generated':
                    return {
                        title: 'Agent plan generated',
                        detail: payload.summary || 'Plan ready for confirmation',
                    };
                case 'vfs_file_updated':
                    return {
                        title: 'VFS updated',
                        detail: payload.path || '',
                    };
                default:
                    return {
                        title: `Event: ${eventType}`,
                        detail: payload.message || '',
                    };
            }
        },

        formatTimestamp(isoString) {
            const date = new Date(isoString);
            return date.toLocaleTimeString();
        },

        clearTimeline() {
            timelineEvents = [];
            this.timelineEvents = [];
            this.renderTimeline();
        },

        // Chat functions
        setChatStatus(state) {
            chatStatus = state;
            this.chatStatus = state;
            if (mockDOM.chatStatusBadge) {
                mockDOM.chatStatusBadge.textContent = state;
            }
        },

        appendChatEntry(role, content) {
            chatMessages.push({
                role,
                content,
                timestamp: new Date().toISOString(),
            });
            if (chatMessages.length > 30) {
                chatMessages.shift();
            }
            this.chatMessages = chatMessages;
            this.renderChatHistory();
        },

        renderChatHistory() {
            const messages = this.chatMessages || chatMessages;
            if (!messages.length) {
                mockDOM.chatHistory.innerHTML = '<div class="empty-state">Select a document and send a command to the agent.</div>';
                return;
            }
            mockDOM.chatHistory.innerHTML = messages
                .map(
                    (entry) => `
                    <div class="chat-entry ${entry.role}">
                        <h6>${entry.role === 'user' ? 'You' : 'Agent'} • ${this.formatTimestamp(entry.timestamp)}</h6>
                        <div>${entry.content}</div>
                    </div>
                `,
                )
                .join('');
        },

        // Room management
        joinDocRoom(fileId) {
            if (!mockSocket || !fileId) {
                return;
            }
            if (joinedRoomId) {
                mockSocket.emit('doc_review:leave', { file_id: joinedRoomId });
            }
            mockSocket.emit('doc_review:join', { token: 'test-token', file_id: fileId });
            joinedRoomId = fileId;
            this.joinedRoomId = joinedRoomId;
            this.setChatStatus('connected');
            timelineEvents = [];
            this.timelineEvents = [];
            this.renderTimeline();
        },

        // Event handlers
        handleDocEvent(eventType, payload = {}) {
            if (payload.file_id && this.joinedRoomId && payload.file_id !== this.joinedRoomId) {
                return;
            }
            this.addTimelineEvent(eventType, payload);
        },

        initSocketHandlers() {
            mockSocket.on('connect', () => this.setChatStatus('connected'));
            mockSocket.on('disconnect', () => this.setChatStatus('offline'));
            mockSocket.on('doc_review:error', (payload) => {
                // In real app: showToast('Stream error', payload?.error || 'Unknown', 'danger');
            });
            ['status', 'log', 'node_started', 'node_completed', 'vfs_file_updated', 'agent_plan_generated'].forEach(
                (eventType) => {
                    mockSocket.on(`doc_review:${eventType}`, (payload) => this.handleDocEvent(eventType, payload));
                },
            );
        },
    };
}

// Test suite
const runner = new TestRunner();
const testFuncs = createTestableFunctions();

// Test 1: Socket connection/disconnection
runner.test('Socket connection updates chat status', () => {
    testFuncs.initSocketHandlers();
    testFuncs.socket.connect();
    runner.assertEqual(testFuncs.chatStatus, 'connected');
    testFuncs.socket.disconnect();
    runner.assertEqual(testFuncs.chatStatus, 'offline');
});

// Test 2: Room join/leave
runner.test('Joining document room emits join event and updates state', () => {
    testFuncs.joinDocRoom('doc-123');
    runner.assertEqual(testFuncs.joinedRoomId, 'doc-123');
    runner.assertEqual(testFuncs.chatStatus, 'connected');
    const emitCalls = testFuncs.socket.emitQueue.filter((e) => e.event === 'doc_review:join');
    runner.assert(emitCalls.length > 0, 'Should emit doc_review:join');
    runner.assertEqual(emitCalls[0].data.file_id, 'doc-123');
});

// Test 3: Room switching
runner.test('Switching documents leaves old room and joins new', () => {
    testFuncs.joinDocRoom('doc-123');
    testFuncs.joinDocRoom('doc-456');
    runner.assertEqual(testFuncs.joinedRoomId, 'doc-456');
    const leaveCalls = testFuncs.socket.emitQueue.filter((e) => e.event === 'doc_review:leave');
    runner.assert(leaveCalls.length > 0, 'Should emit leave for old room');
    runner.assertEqual(leaveCalls[leaveCalls.length - 1].data.file_id, 'doc-123');
});

// Test 4: Timeline event handling
runner.test('node_started event adds timeline entry', () => {
    testFuncs.clearTimeline();
    testFuncs.joinDocRoom('doc-123');
    testFuncs.handleDocEvent('node_started', {
        file_id: 'doc-123',
        node: 'phase1_doc_summary',
        summary: 'Analyzing document',
    });
    runner.assert(testFuncs.timelineEvents.length === 1, `Should have one timeline event, got ${testFuncs.timelineEvents.length}`);
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'Node started: phase1_doc_summary');
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'Analyzing document');
});

// Test 5: node_completed event
runner.test('node_completed event adds timeline entry with status', () => {
    testFuncs.clearTimeline();
    testFuncs.joinDocRoom('doc-123');
    testFuncs.handleDocEvent('node_completed', {
        file_id: 'doc-123',
        node: 'phase1_doc_summary',
        status: 'success',
        summary: 'Document analyzed',
    });
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'Node completed: phase1_doc_summary');
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'success');
});

// Test 6: agent_plan_generated event
runner.test('agent_plan_generated event renders plan summary', () => {
    testFuncs.clearTimeline();
    testFuncs.joinDocRoom('doc-123');
    testFuncs.handleDocEvent('agent_plan_generated', {
        file_id: 'doc-123',
        summary: 'Plan to run Phase 2',
    });
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'Agent plan generated');
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'Plan to run Phase 2');
});

// Test 7: vfs_file_updated event
runner.test('vfs_file_updated event shows file path', () => {
    testFuncs.clearTimeline();
    testFuncs.joinDocRoom('doc-123');
    testFuncs.handleDocEvent('vfs_file_updated', {
        file_id: 'doc-123',
        path: '/phase1/doc_summary.json',
    });
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'VFS updated');
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, '/phase1/doc_summary.json');
});

// Test 8: Timeline event filtering by file_id
runner.test('Events filtered by file_id when room joined', () => {
    testFuncs.joinDocRoom('doc-123');
    testFuncs.clearTimeline();
    // Event for different file_id should be ignored
    testFuncs.handleDocEvent('node_started', {
        file_id: 'doc-999',
        node: 'other_node',
    });
    runner.assertEqual(testFuncs.timelineEvents.length, 0, 'Should ignore events for other documents');
    // Event for current file_id should be added
    testFuncs.handleDocEvent('node_started', {
        file_id: 'doc-123',
        node: 'phase1_doc_summary',
    });
    runner.assertEqual(testFuncs.timelineEvents.length, 1, 'Should add events for current document');
});

// Test 9: Timeline limit (max 40 events)
runner.test('Timeline limits to 40 events', () => {
    testFuncs.clearTimeline();
    for (let i = 0; i < 45; i++) {
        testFuncs.addTimelineEvent('node_started', { node: `node_${i}` });
    }
    runner.assertEqual(testFuncs.timelineEvents.length, 40, 'Should limit to 40 events');
});

// Test 10: Chat entry rendering
runner.test('Chat entries render user and agent messages', () => {
    testFuncs.chatMessages = [];
    testFuncs.appendChatEntry('user', 'Run phase 2');
    testFuncs.appendChatEntry('agent', 'Plan generated');
    runner.assert(testFuncs.chatMessages.length === 2);
    runner.assertContains(testFuncs.dom.chatHistory.innerHTML, 'You');
    runner.assertContains(testFuncs.dom.chatHistory.innerHTML, 'Agent');
    runner.assertContains(testFuncs.dom.chatHistory.innerHTML, 'Run phase 2');
    runner.assertContains(testFuncs.dom.chatHistory.innerHTML, 'Plan generated');
});

// Test 11: Chat message limit (max 30)
runner.test('Chat history limits to 30 messages', () => {
    testFuncs.chatMessages = [];
    for (let i = 0; i < 35; i++) {
        testFuncs.appendChatEntry('user', `Message ${i}`);
    }
    runner.assertEqual(testFuncs.chatMessages.length, 30, 'Should limit to 30 messages');
});

// Test 12: Empty timeline rendering
runner.test('Empty timeline shows placeholder', () => {
    testFuncs.clearTimeline();
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'Live workflow events will appear here');
});

// Test 13: Empty chat rendering
runner.test('Empty chat shows placeholder', () => {
    testFuncs.chatMessages = [];
    testFuncs.renderChatHistory();
    runner.assertContains(testFuncs.dom.chatHistory.innerHTML, 'Select a document and send a command');
});

// Test 14: Multiple event types in timeline
runner.test('Timeline handles multiple event types correctly', () => {
    testFuncs.clearTimeline();
    testFuncs.handleDocEvent('node_started', { node: 'phase1', file_id: 'doc-123' });
    testFuncs.handleDocEvent('node_completed', { node: 'phase1', file_id: 'doc-123' });
    testFuncs.handleDocEvent('vfs_file_updated', { path: '/test.json', file_id: 'doc-123' });
    runner.assertEqual(testFuncs.timelineEvents.length, 3);
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'Node started');
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'Node completed');
    runner.assertContains(testFuncs.dom.timelineFeed.innerHTML, 'VFS updated');
});

// Test 15: Socket event listener registration
runner.test('Socket event listeners registered for all doc_review events', () => {
    testFuncs.socket.clear();
    testFuncs.initSocketHandlers();
    const expectedEvents = ['connect', 'disconnect', 'doc_review:error', 'doc_review:status', 'doc_review:log', 'doc_review:node_started', 'doc_review:node_completed', 'doc_review:vfs_file_updated', 'doc_review:agent_plan_generated'];
    expectedEvents.forEach((event) => {
        runner.assert(testFuncs.socket.listeners[event] && testFuncs.socket.listeners[event].length > 0, `Should have listener for ${event}`);
    });
});

// Run tests
if (typeof module !== 'undefined' && module.exports) {
    // Node.js
    module.exports = { TestRunner, MockSocketIO, createTestableFunctions };
    runner.run().then((success) => {
        process.exit(success ? 0 : 1);
    });
} else {
    // Browser
    runner.run();
}

