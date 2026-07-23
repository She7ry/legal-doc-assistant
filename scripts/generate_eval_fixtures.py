from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import wrap

from ai.config.settings import settings
from ai.rag.evaluation.constants import DEFAULT_REFUSAL_TERMS
from ai.rag.ingestion.loader import file_sha256, load_documents
from ai.rag.retrieval.vector_store import (
    INGESTION_CHUNK_SEPARATORS,
    split_documents_for_ingestion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "data" / "eval" / "fixtures"
DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_dataset.json"


@dataclass(frozen=True)
class EvalDocument:
    file_name: str
    pages: list[list[str]]


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    answer_type: str
    gold_answer: str
    markers: tuple[str, ...] = ()
    required_answer_terms: tuple[str, ...] = ()
    forbidden_answer_terms: tuple[str, ...] = ()
    required_refusal_terms: tuple[str, ...] = ()


DOCUMENTS = [
    EvalDocument(
        file_name="sample_chinese_purchase_contract.pdf",
        pages=[
            [
                "华辰设备采购合同",
                "文档编号：EVAL-ZH-CONTRACT-008",
                "",
                "第一条 合同主体与通知",
                "Marker: EVAL-ZH-C-1.1.",
                "甲方（采购方）：华辰智能制造有限公司。",
                "乙方（供应方）：远海精密设备有限公司。",
                "双方确认合同首页所列注册地址、电子邮箱和联系人为有效通知信息。",
                "涉及交付、验收、付款或违约的通知应以电子邮件并辅以专人送达或快递方式发出。",
                "联系人或送达信息发生变化的，变更方应至少提前3个工作日书面通知对方。",
                "未及时通知造成文件延误或无法送达的，相应不利后果由变更方承担。",
                "",
                "第二条 合同价款",
                "Marker: EVAL-ZH-C-2.1.",
                "合同总价为人民币1,000,000元（大写：壹佰万元整），该价款已经包含增值税。",
                "合同价款还包括设备包装、运输、运输保险、卸货、安装调试、培训和技术资料费用。",
                "除双方签署书面变更单外，乙方不得以原材料或人工成本上涨为由追加费用。",
                "乙方应按照甲方确认的开票信息开具与当期应付款金额一致的增值税专用发票。",
                "因乙方原因发生的返工、补运或重复检验费用已经包含在合同总价中。",
                "",
                "第三条 付款期限",
                "Marker: EVAL-ZH-C-3.1.",
                "甲方应在收到乙方合法有效的增值税专用发票后10个工作日内支付合同价款。",
                "发票信息有误的，付款期限自乙方重新提交正确发票之日起计算。",
                "甲方对部分金额有合理异议的，可以暂缓支付争议部分，但应按期支付无争议部分。",
                "乙方收款账户应与合同约定一致，账户变更须至少提前5个工作日书面通知甲方。",
                "甲方付款不视为放弃验收权，也不免除乙方对质量缺陷承担的责任。",
            ],
            [
                "华辰设备采购合同",
                "文档编号：EVAL-ZH-CONTRACT-008",
                "",
                "第四条 交付与验收",
                "Marker: EVAL-ZH-C-4.1.",
                "乙方应于2026年9月30日前将全部设备交付至甲方上海工厂。",
                "甲方应在设备到货后5个工作日内完成外观和数量验收。",
                "乙方应至少提前3个工作日提交到货计划，并随货提供装箱单、合格证和操作手册。",
                "设备完成安装调试后，甲方在连续试运行15日内进行性能验收。",
                "验收发现短缺、破损或性能不合格的，乙方应在甲方指定期限内免费补足、修复或更换。",
                "甲方签收运输单据或未在初验期内提出异议，不视为对隐蔽缺陷或性能指标的最终认可。",
                "",
                "第五条 逾期交付",
                "Marker: EVAL-ZH-C-5.1.",
                "乙方每逾期一日，应按迟延交付批次价款的0.3%支付违约金。",
                "逾期交付违约金累计不超过迟延交付批次价款的8%。",
                "因甲方书面确认的交付条件未具备导致延期的，相应期限可以按实际受影响天数顺延。",
                "乙方预计无法按期交付时，应立即提交原因、影响范围和赶工计划，不因此免除违约责任。",
                "甲方接受迟延交付或收取部分设备，不构成对违约金及其他救济权利的放弃。",
                "逾期超过20日的，甲方可以另行采购替代设备，并要求乙方承担合理差价和直接损失。",
                "",
                "第六条 违约解除",
                "Marker: EVAL-ZH-C-6.1.",
                "一方严重违约，经守约方书面催告后15日内仍未改正的，守约方可以解除合同。",
                "严重违约包括持续交付不合格设备、拒绝履行主要义务、逾期交付超过20日或违反保密义务。",
                "违约方进入破产清算、停止营业或明确表示不再履行主要义务的，守约方可以立即解除合同。",
                "解除通知送达违约方时生效，乙方应在5个工作日内返还尚未对应合格交付的预付款。",
                "合同解除不影响守约方请求违约金、损害赔偿以及履行保密和知识产权义务的权利。",
                "",
                "第七条 适用法律与争议解决",
                "Marker: EVAL-ZH-C-7.1.",
                "本合同适用中华人民共和国法律。因本合同产生的争议提交上海仲裁委员会仲裁。",
                "争议发生后，双方应先由项目负责人进行不少于10个工作日的善意协商。",
                "协商未果的，任何一方均可按照申请仲裁时有效的仲裁规则提起仲裁。",
                "仲裁语言为中文，仲裁裁决为终局，对双方均有约束力。",
                "争议处理期间，除争议事项外，双方应继续履行不受影响的合同义务。",
            ],
            [
                "华辰设备采购合同",
                "文档编号：EVAL-ZH-CONTRACT-008",
                "",
                "第八条 质量保证与售后服务",
                "Marker: EVAL-ZH-C-8.1.",
                "设备质量保证期为最终验收合格之日起24个月。",
                "保修期内发生故障的，乙方应在收到通知后4小时内响应，并在48小时内到达现场处理。",
                "因设计、材料或制造缺陷产生的检测、维修、更换、运输和差旅费用由乙方承担。",
                "维修或更换部件的保修期为原剩余保修期与修复后6个月两者中的较长期限。",
                "同一关键故障累计发生三次且仍影响正常使用的，甲方有权要求更换整机。",
                "保修期届满后，乙方仍应提供不少于5年的备件供应和有偿技术支持。",
                "",
                "第九条 知识产权与保密",
                "Marker: EVAL-ZH-C-9.1.",
                "乙方保证有权提供合同设备、软件和技术资料，并授权甲方为使用和维护设备进行必要复制。",
                "第三方主张侵权时，乙方应负责处理并承担费用，必要时取得授权、更换方案或退还受影响价款。",
                "双方对在履约过程中获悉的非公开技术、价格、经营和客户信息承担保密义务。",
                "接收方只能为履行本合同使用保密信息，并应采取不低于保护自身同类信息的合理措施。",
                "一般保密义务在合同终止后持续5年，依法构成商业秘密的信息在其秘密性存续期间持续受保护。",
                "依法必须披露时，接收方应在法律允许范围内提前通知披露方，并将披露范围控制在最低限度。",
            ],
            [
                "华辰设备采购合同",
                "文档编号：EVAL-ZH-CONTRACT-008",
                "",
                "第十条 不可抗力",
                "Marker: EVAL-ZH-C-10.1.",
                "不可抗力是双方无法合理预见、避免且克服，并直接妨碍合同履行的客观事件。",
                "受影响方应在事件发生后5日内书面通知对方，并在10日内提交事件和影响程度的证明。",
                "受影响方应采取合理措施减少损失，并持续报告预计恢复时间和替代履行方案。",
                "履行期限仅在实际受影响范围内顺延；资金不足或可合理避免的人员短缺不构成不可抗力。",
                "不可抗力连续超过30日的，任何一方均可书面解除受影响部分，双方按已合格履行部分结算。",
                "不可抗力不免除事件发生前已经到期的付款义务。",
                "",
                "第十一条 记录保存与审计",
                "Marker: EVAL-ZH-C-11.1.",
                "乙方应自最终付款之日起保存采购、生产、检验、交付和发票记录至少5年。",
                "甲方每个自然年度可在提前10个工作日书面通知后，对与本合同直接相关的记录审计一次。",
                "审计应在正常工作时间进行，不得不合理干扰乙方经营或查阅与本合同无关的信息。",
                "甲方及其审计人员应对审计中获得的信息保密，但法律法规另有要求的除外。",
                "审计发现多收金额超过当期受查金额2%的，乙方应在10个工作日内退款并承担合理审计费用。",
                "电子记录应保持完整、可检索并能够说明审批人、操作时间和修改历史。",
            ],
        ],
    ),
    EvalDocument(
        file_name="sample_chinese_privacy_policy.pdf",
        pages=[
            [
                "个人信息处理管理制度",
                "文档编号：EVAL-ZH-POLICY-009",
                "",
                "一、适用范围",
                "Marker: EVAL-ZH-P-1.1.",
                "本制度适用于公司员工、承包商和受托服务商处理客户及员工个人信息的活动。",
                "制度覆盖个人信息的收集、使用、存储、传输、共享、归档和删除等全部处理环节。",
                "各业务部门负责人是本部门个人信息处理活动的责任人，应确保处理目的明确并遵循最小必要原则。",
                "新系统或处理目的发生重大变化前，业务部门应会同数据保护负责人完成个人信息保护影响评估。",
                "接触个人信息的人员在取得权限前应完成培训，此后每年至少复训一次。",
                "",
                "二、个人信息跨境提供",
                "Marker: EVAL-ZH-P-2.1.",
                "向境外接收方提供个人信息前，业务部门必须取得法务部和数据保护负责人的书面批准。",
                "业务部门还应完成适用的数据出境合规程序，并记录接收方、数据类别、目的和保存期限。",
                "依法需要单独同意、标准合同、认证或安全评估的，应在传输开始前完成相应手续。",
                "信息安全部应验证传输加密、接收方身份认证和下载权限，禁止使用未经批准的个人网盘或邮箱。",
                "跨境提供安排至少每年复核一次；接收方、数据类别或处理目的变化时应重新审批。",
                "",
                "三、安全事件报告",
                "Marker: EVAL-ZH-P-3.1.",
                "发现涉及个人信息的疑似或确认安全事件后，应在24小时内报告信息安全部和数据保护负责人。",
                "报告应说明事件时间、影响范围、涉及的数据类别和已经采取的控制措施。",
                "发现人应立即停止未经授权的访问或传输，并保留日志、邮件和设备镜像等证据。",
                "信息安全部负责组织隔离、调查和恢复，数据保护负责人判断是否需要通知监管机构或个人。",
                "事件关闭后10个工作日内应完成复盘，明确根因、整改责任人和计划完成日期。",
            ],
            [
                "个人信息处理管理制度",
                "文档编号：EVAL-ZH-POLICY-009",
                "",
                "四、保存与删除",
                "Marker: EVAL-ZH-P-4.1.",
                "个人信息处理活动记录应至少保存3年。",
                "处理目的实现或业务关系终止后，应在30日内删除或者匿名化相关个人信息；法律另有保存要求的除外。",
                "因诉讼、调查或法定义务需要继续保存的，应暂停常规删除并记录保存依据、范围和到期时间。",
                "暂停删除期间，相关信息应隔离存储、限制访问，不得继续用于原业务目的之外的活动。",
                "删除或匿名化完成后，执行人应记录数据范围、处理方式、完成时间和复核人。",
                "",
                "五、访问控制",
                "Marker: EVAL-ZH-P-5.1.",
                "管理员账户、远程访问以及存储敏感个人信息的系统必须启用多因素认证。",
                "个人信息访问权限应当每季度复核一次，并及时撤销不再需要的权限。",
                "权限申请应说明业务目的、数据范围和期限，并由直属负责人及数据责任人批准。",
                "员工离职或调离岗位后，相关访问权限应在1个工作日内撤销或调整。",
                "查询、导出、修改和删除个人信息的操作日志应至少保存1年，并防止普通用户篡改。",
                "",
                "六、例外审批",
                "Marker: EVAL-ZH-P-6.1.",
                "任何偏离本制度的例外必须由信息安全负责人和数据保护负责人书面批准，并明确有效期限。",
                "例外申请应说明业务必要性、涉及的数据、风险评估结果和拟采取的补偿性控制。",
                "单次例外有效期最长为6个月；到期仍需继续的，应重新评估并办理审批。",
                "例外责任人应至少每月确认补偿性控制仍然有效，条件变化时立即报告审批人。",
                "例外不得减损个人依法享有的权利，也不得免除法律法规规定的强制义务。",
            ],
            [
                "个人信息处理管理制度",
                "文档编号：EVAL-ZH-POLICY-009",
                "",
                "七、受托处理与供应商管理",
                "Marker: EVAL-ZH-P-7.1.",
                "委托供应商处理个人信息前，业务部门应完成隐私和安全尽职调查，并确认其具备相应保护能力。",
                "双方合同应明确处理目的、期限、数据类别、安全措施、事件报告、协助义务和终止后的处置方式。",
                "供应商只能按照公司的书面指示处理个人信息，未经书面批准不得转委托其他处理者。",
                "高风险供应商至少每年重新评估一次，发现重大缺陷时应制定限期整改计划或停止传输数据。",
                "委托结束后，供应商应在30日内返还或删除个人信息，并提供经授权人员签署的删除证明。",
                "业务部门应保存尽职调查、合同、评估和删除证明，保存期限不得短于合作终止后3年。",
                "",
                "八、个人权利请求",
                "Marker: EVAL-ZH-P-8.1.",
                "公司通过客服渠道和隐私邮箱接收查阅、更正、删除、复制或撤回同意等个人权利请求。",
                "受理人员应核验请求人身份，并仅收集完成核验所必需的信息，避免向冒名者披露数据。",
                "除法律另有规定外，公司应在15个工作日内完成请求并向请求人反馈处理结果。",
                "情况复杂确需延期的，可以延长一次且不超过15个工作日，并在原期限届满前说明理由。",
                "无法满足请求时，应说明适用的法律依据、拒绝理由以及投诉或申诉渠道。",
                "请求内容、核验材料、处理决定和回复记录应至少保存3年。",
            ],
            [
                "个人信息处理管理制度",
                "文档编号：EVAL-ZH-POLICY-009",
                "",
                "九、收集与使用规范",
                "Marker: EVAL-ZH-P-9.1.",
                "收集个人信息前应确定具体、明确且合理的处理目的，并将字段控制在实现目的所必需的范围。",
                "业务页面应以清晰方式告知处理者身份、信息种类、处理目的、保存期限和权利行使方式。",
                "处理敏感个人信息应具有特定目的和充分必要性，依法取得单独同意并采取更严格的保护措施。",
                "拟将个人信息用于原目的之外的新用途时，应重新评估合法性并履行必要的告知或同意义务。",
                "开发、测试和演示环境原则上使用合成或脱敏数据，不得直接复制生产环境完整个人信息。",
                "通过自动化决策对个人权益产生重大影响时，应提供透明说明和便捷的人工复核渠道。",
                "",
                "十、备份数据与删除核验",
                "Marker: EVAL-ZH-P-10.1.",
                "进入删除流程的个人信息应同步从检索索引、缓存、分析副本和日常导出文件中清除。",
                "备份中的待删除信息应被标记并限制使用，仅可在灾难恢复所必需时随备份恢复。",
                "恢复备份后，应重新执行删除任务；待删除信息最迟在90日内随备份轮换永久清除。",
                "数据责任人每月应抽查不少于10%的已完成删除任务，核对主系统、副本和日志记录。",
                "删除失败或发现残留副本时，应在2个工作日内升级报告，并在10个工作日内完成整改。",
                "删除核验记录应包含任务编号、系统范围、抽查结果、异常处置和复核人。",
            ],
        ],
    ),
]


CASES = [
    EvalCase(
        id="eval_049_zh_contract_total_price",
        question="采购合同的含税总价是多少？",
        answer_type="answerable",
        gold_answer="合同含税总价为人民币1,000,000元，即壹佰万元整。",
        markers=("EVAL-ZH-C-2.1",),
        required_answer_terms=("人民币1,000,000元", "壹佰万元"),
        forbidden_answer_terms=("80万元",),
    ),
    EvalCase(
        id="eval_050_zh_contract_payment_period",
        question="甲方收到正确发票后应在多久内付款？",
        answer_type="answerable",
        gold_answer="甲方应在收到合法有效的增值税专用发票后10个工作日内付款。",
        markers=("EVAL-ZH-C-3.1",),
        required_answer_terms=("10个工作日", "增值税专用发票"),
        forbidden_answer_terms=("10个自然日",),
    ),
    EvalCase(
        id="eval_051_zh_contract_delivery_date",
        question="乙方最迟应在什么日期交付全部设备？",
        answer_type="answerable",
        gold_answer="乙方最迟应于2026年9月30日交付全部设备。",
        markers=("EVAL-ZH-C-4.1",),
        required_answer_terms=("2026年9月30日",),
        forbidden_answer_terms=("2026年10月",),
    ),
    EvalCase(
        id="eval_052_zh_contract_acceptance_period",
        question="设备到货后，甲方有几个工作日完成验收？",
        answer_type="answerable",
        gold_answer="甲方应在设备到货后5个工作日内完成外观和数量验收。",
        markers=("EVAL-ZH-C-4.1",),
        required_answer_terms=("5个工作日", "验收"),
        forbidden_answer_terms=("10个工作日",),
    ),
    EvalCase(
        id="eval_053_zh_contract_daily_damages",
        question="逾期交付违约金按每日什么比例计算？",
        answer_type="answerable",
        gold_answer="每逾期一日，按迟延交付批次价款的0.3%计算违约金。",
        markers=("EVAL-ZH-C-5.1",),
        required_answer_terms=("0.3%", "迟延交付批次价款"),
        forbidden_answer_terms=("0.5%",),
    ),
    EvalCase(
        id="eval_054_zh_contract_damages_cap",
        question="逾期交付违约金的累计上限是多少？",
        answer_type="answerable",
        gold_answer="累计违约金不超过迟延交付批次价款的8%。",
        markers=("EVAL-ZH-C-5.1",),
        required_answer_terms=("8%", "迟延交付批次价款"),
        forbidden_answer_terms=("10%",),
    ),
    EvalCase(
        id="eval_055_zh_contract_cure_period",
        question="严重违约后经过多久仍未改正，守约方可以解除合同？",
        answer_type="answerable",
        gold_answer="经书面催告后15日内仍未改正的，守约方可以解除合同。",
        markers=("EVAL-ZH-C-6.1",),
        required_answer_terms=("15日", "书面催告"),
        forbidden_answer_terms=("30日",),
    ),
    EvalCase(
        id="eval_056_zh_contract_dispute_resolution",
        question="合同适用什么法律，争议提交哪里处理？",
        answer_type="answerable",
        gold_answer="合同适用中华人民共和国法律，争议提交上海仲裁委员会仲裁。",
        markers=("EVAL-ZH-C-7.1",),
        required_answer_terms=("中华人民共和国法律", "上海仲裁委员会"),
        forbidden_answer_terms=("上海法院",),
    ),
    EvalCase(
        id="eval_057_zh_contract_insurance_refusal",
        question="合同是否要求乙方购买产品责任保险？",
        answer_type="unanswerable",
        gold_answer="索引合同未说明乙方必须购买产品责任保险。",
        required_refusal_terms=("未说明",),
    ),
    EvalCase(
        id="eval_058_zh_policy_incident_notice",
        question="发现个人信息安全事件后应在多久内报告？",
        answer_type="answerable",
        gold_answer="发现疑似或确认的个人信息安全事件后，应在24小时内报告。",
        markers=("EVAL-ZH-P-3.1",),
        required_answer_terms=("24小时", "报告"),
        forbidden_answer_terms=("48小时",),
    ),
    EvalCase(
        id="eval_059_zh_policy_record_retention",
        question="个人信息处理活动记录至少保存几年？",
        answer_type="answerable",
        gold_answer="个人信息处理活动记录至少保存3年。",
        markers=("EVAL-ZH-P-4.1",),
        required_answer_terms=("3年",),
        forbidden_answer_terms=("5年",),
    ),
    EvalCase(
        id="eval_060_zh_policy_deletion_period",
        question="处理目的实现后，应在多少日内删除或匿名化个人信息？",
        answer_type="answerable",
        gold_answer="处理目的实现或业务关系终止后，应在30日内删除或匿名化个人信息。",
        markers=("EVAL-ZH-P-4.1",),
        required_answer_terms=("30日", "删除", "匿名化"),
        forbidden_answer_terms=("45日",),
    ),
    EvalCase(
        id="eval_061_zh_policy_cross_border_approval",
        question="向境外接收方提供个人信息前需要谁书面批准？",
        answer_type="answerable",
        gold_answer="需要取得法务部和数据保护负责人的书面批准。",
        markers=("EVAL-ZH-P-2.1",),
        required_answer_terms=("法务部", "数据保护负责人", "书面批准"),
        forbidden_answer_terms=("信息安全部单独批准",),
    ),
    EvalCase(
        id="eval_062_zh_policy_mfa_scope",
        question="哪些访问场景必须启用多因素认证？",
        answer_type="answerable",
        gold_answer="管理员账户、远程访问和存储敏感个人信息的系统必须启用多因素认证。",
        markers=("EVAL-ZH-P-5.1",),
        required_answer_terms=("管理员账户", "远程访问", "敏感个人信息"),
        forbidden_answer_terms=("所有普通账户",),
    ),
    EvalCase(
        id="eval_063_zh_policy_biometric_refusal",
        question="制度规定的人脸识别数据保存期限是多少？",
        answer_type="unanswerable",
        gold_answer="索引制度未说明人脸识别数据的具体保存期限。",
        required_refusal_terms=("未说明",),
    ),
    EvalCase(
        id="eval_067_zh_contract_contact_change_notice",
        question="合同联系人或送达信息发生变化时，应提前多久通知对方？",
        answer_type="answerable",
        gold_answer="变更方应至少提前3个工作日书面通知对方。",
        markers=("EVAL-ZH-C-1.1",),
        required_answer_terms=("3个工作日", "书面通知"),
        forbidden_answer_terms=("5个工作日",),
    ),
    EvalCase(
        id="eval_068_zh_contract_warranty_period",
        question="设备质量保证期从何时起算，共持续多久？",
        answer_type="answerable",
        gold_answer="质量保证期从最终验收合格之日起计算，共24个月。",
        markers=("EVAL-ZH-C-8.1",),
        required_answer_terms=("最终验收合格", "24个月"),
        forbidden_answer_terms=("12个月",),
    ),
    EvalCase(
        id="eval_069_zh_contract_service_response",
        question="保修期内发生故障后，乙方的响应和到场时限分别是多少？",
        answer_type="answerable",
        gold_answer="乙方应在收到通知后4小时内响应，并在48小时内到达现场处理。",
        markers=("EVAL-ZH-C-8.1",),
        required_answer_terms=("4小时", "48小时"),
        forbidden_answer_terms=("24小时内到场",),
    ),
    EvalCase(
        id="eval_070_zh_contract_confidentiality_term",
        question="合同终止后，一般保密义务和商业秘密保护分别持续多久？",
        answer_type="answerable",
        gold_answer="一般保密义务持续5年，商业秘密在其秘密性存续期间持续受保护。",
        markers=("EVAL-ZH-C-9.1",),
        required_answer_terms=("5年", "秘密性存续期间"),
        forbidden_answer_terms=("3年",),
    ),
    EvalCase(
        id="eval_071_zh_contract_force_majeure_deadlines",
        question="不可抗力发生后需要在多久内通知和提交证明，持续多久可解除受影响部分？",
        answer_type="answerable",
        gold_answer="受影响方应在5日内通知、10日内提交证明；不可抗力连续超过30日时，任何一方可解除受影响部分。",
        markers=("EVAL-ZH-C-10.1",),
        required_answer_terms=("5日", "10日", "30日"),
        forbidden_answer_terms=("60日",),
    ),
    EvalCase(
        id="eval_072_zh_contract_audit_records",
        question="乙方应保存合同记录多久，甲方审计需要提前多久通知？",
        answer_type="answerable",
        gold_answer="乙方应自最终付款之日起至少保存记录5年；甲方审计应提前10个工作日书面通知。",
        markers=("EVAL-ZH-C-11.1",),
        required_answer_terms=("5年", "10个工作日"),
        forbidden_answer_terms=("3年",),
    ),
    EvalCase(
        id="eval_073_zh_policy_training_frequency",
        question="接触个人信息的人员应在何时接受培训，之后多久复训一次？",
        answer_type="answerable",
        gold_answer="人员应在取得权限前完成培训，此后每年至少复训一次。",
        markers=("EVAL-ZH-P-1.1",),
        required_answer_terms=("取得权限前", "每年"),
        forbidden_answer_terms=("每季度",),
    ),
    EvalCase(
        id="eval_074_zh_policy_cross_border_review",
        question="个人信息跨境提供安排多久复核一次，哪些变化会触发重新审批？",
        answer_type="answerable",
        gold_answer="跨境提供安排至少每年复核一次；接收方、数据类别或处理目的变化时应重新审批。",
        markers=("EVAL-ZH-P-2.1",),
        required_answer_terms=("每年", "接收方", "数据类别", "处理目的"),
        forbidden_answer_terms=("每五年",),
    ),
    EvalCase(
        id="eval_075_zh_policy_offboarding_access",
        question="员工离职或调岗后，个人信息访问权限应在多久内撤销或调整？",
        answer_type="answerable",
        gold_answer="相关访问权限应在1个工作日内撤销或调整。",
        markers=("EVAL-ZH-P-5.1",),
        required_answer_terms=("1个工作日", "撤销", "调整"),
        forbidden_answer_terms=("5个工作日",),
    ),
    EvalCase(
        id="eval_076_zh_policy_exception_period",
        question="制度例外由谁批准，单次最长有效期是多少？",
        answer_type="answerable",
        gold_answer="例外须由信息安全负责人和数据保护负责人书面批准，单次最长有效期为6个月。",
        markers=("EVAL-ZH-P-6.1",),
        required_answer_terms=("信息安全负责人", "数据保护负责人", "6个月"),
        forbidden_answer_terms=("12个月",),
    ),
    EvalCase(
        id="eval_077_zh_policy_vendor_deletion",
        question="委托处理结束后，供应商应在多久内处置个人信息并提供什么材料？",
        answer_type="answerable",
        gold_answer="供应商应在30日内返还或删除个人信息，并提供经授权人员签署的删除证明。",
        markers=("EVAL-ZH-P-7.1",),
        required_answer_terms=("30日", "返还", "删除证明"),
        forbidden_answer_terms=("90日",),
    ),
    EvalCase(
        id="eval_078_zh_policy_rights_request",
        question="公司通常应在多久内完成个人权利请求，复杂情形最多可延长多久？",
        answer_type="answerable",
        gold_answer="公司通常应在15个工作日内完成请求；复杂情形可延长一次，最长再延长15个工作日。",
        markers=("EVAL-ZH-P-8.1",),
        required_answer_terms=("15个工作日", "延长一次"),
        forbidden_answer_terms=("30个自然日",),
    ),
    EvalCase(
        id="eval_079_zh_policy_sensitive_information",
        question="制度对处理敏感个人信息提出了哪些额外要求？",
        answer_type="answerable",
        gold_answer="处理敏感个人信息应具有特定目的和充分必要性，依法取得单独同意并采取更严格的保护措施。",
        markers=("EVAL-ZH-P-9.1",),
        required_answer_terms=("特定目的", "充分必要性", "单独同意"),
        forbidden_answer_terms=("无需同意",),
    ),
    EvalCase(
        id="eval_080_zh_policy_backup_deletion",
        question="待删除的个人信息最迟应在多久内从备份中永久清除？",
        answer_type="answerable",
        gold_answer="待删除信息最迟应在90日内随备份轮换永久清除。",
        markers=("EVAL-ZH-P-10.1",),
        required_answer_terms=("90日", "备份轮换", "永久清除"),
        forbidden_answer_terms=("30日",),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the starter RAG eval fixtures.")
    parser.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    document_paths = []
    with TemporaryDirectory(dir=FIXTURE_DIR) as generated_dir:
        for document in DOCUMENTS:
            path = FIXTURE_DIR / document.file_name
            generated_path = Path(generated_dir) / document.file_name
            _write_simple_pdf(generated_path, document.pages)
            generated_bytes = generated_path.read_bytes()
            if not path.exists() or path.read_bytes() != generated_bytes:
                path.write_bytes(generated_bytes)
            document_paths.append(path)

    marker_sources = _find_marker_sources(
        document_paths,
        sorted({marker for case in CASES for marker in case.markers}),
    )

    dataset = {
        "version": "0.4",
        "description": (
            "中文合成法律 RAG 评测集。文档均为虚构内容，结构参考常见合同、制度与合规材料。"
        ),
        "default_refusal_terms": list(DEFAULT_REFUSAL_TERMS),
        "chunking": _chunking_metadata(),
        "documents": [
            {
                "file_name": path.name,
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "file_id": file_sha256(path),
            }
            for path in document_paths
        ],
        "cases": [_case_to_dict(case, marker_sources) for case in CASES],
    }

    DATASET_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(document_paths)} PDF fixtures to {FIXTURE_DIR}")
    print(f"Wrote {len(CASES)} eval cases to {DATASET_PATH}")


def _case_to_dict(case: EvalCase, marker_sources: dict[str, dict[str, object]]) -> dict[str, object]:
    data: dict[str, object] = {
        "id": case.id,
        "question": case.question,
        "answer_type": case.answer_type,
        "category": _case_category(case),
        "tags": _case_tags(case),
        "gold_answer": case.gold_answer,
        "gold_sources": [marker_sources[marker] for marker in case.markers],
    }
    if case.answer_type == "unanswerable":
        if case.required_refusal_terms:
            data["required_refusal_terms"] = list(case.required_refusal_terms)
        return data
    if case.required_answer_terms:
        data["required_answer_terms"] = list(case.required_answer_terms)
    if case.forbidden_answer_terms:
        data["forbidden_answer_terms"] = list(case.forbidden_answer_terms)
    return data


def _case_category(case: EvalCase) -> str:
    if len(case.markers) > 1 or "cross_doc" in case.id:
        return "cross_document"
    if "_zh_contract_" in case.id:
        return "chinese_purchase_contract"
    if "_zh_policy_" in case.id:
        return "chinese_data_policy"
    if "_procurement_" in case.id:
        return "procurement_policy"
    if "_security_" in case.id:
        return "information_security_policy"
    if "_dpa_" in case.id:
        return "data_processing_addendum"
    if "_saas_" in case.id:
        return "saas_msa"
    if "_hr_" in case.id:
        return "employee_handbook"
    if "_nda_" in case.id:
        return "mutual_nda"
    return "supply_contract"


def _case_tags(case: EvalCase) -> list[str]:
    tags = [case.answer_type, _case_category(case)]
    if _contains_cjk(case.question):
        tags.append("chinese_query")
    if _case_category(case).startswith("chinese_") or any(
        marker.startswith("EVAL-ZH-") for marker in case.markers
    ):
        tags.append("chinese_document")
    if len(case.markers) > 1 or "cross_doc" in case.id:
        tags.append("cross_document")
    return tags


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _find_marker_sources(
    pdf_paths: list[Path],
    markers: list[str],
) -> dict[str, dict[str, object]]:
    remaining = set(markers)
    found: dict[str, dict[str, object]] = {}
    for pdf_path in pdf_paths:
        chunks = _split_like_ingestion(pdf_path)
        for chunk_id, chunk in enumerate(chunks):
            text = chunk.page_content or ""
            for marker in list(remaining):
                if marker not in text:
                    continue
                metadata = chunk.metadata or {}
                found[marker] = {
                    "file_name": pdf_path.name,
                    "page": metadata.get("page"),
                    "chunk_id": chunk_id,
                    "marker": marker,
                }
                remaining.remove(marker)
        if not remaining:
            break

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise RuntimeError(f"Could not find generated eval markers in chunks: {missing}")
    return found


def _split_like_ingestion(pdf_path: Path):
    return split_documents_for_ingestion(load_documents(pdf_path))


def _chunking_metadata() -> dict[str, object]:
    payload = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "separators": list(INGESTION_CHUNK_SEPARATORS),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return {**payload, "config_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def _write_simple_pdf(path: Path, pages: list[list[str]]) -> None:
    if any(_contains_cjk(line) for page in pages for line in page) and _write_reportlab_pdf(
        path, pages
    ):
        return
    _write_minimal_pdf(path, pages)


def _write_reportlab_pdf(path: Path, pages: list[list[str]]) -> bool:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfgen import canvas
    except ImportError:
        return False

    font_name = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    document = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    _, page_height = letter
    for lines in pages:
        y = page_height - 42
        document.setFont(font_name, 10)
        for raw_line in lines:
            wrapped_lines = wrap(raw_line, width=96) if raw_line else [""]
            for line in wrapped_lines:
                if y < 42:
                    document.showPage()
                    document.setFont(font_name, 10)
                    y = page_height - 42
                document.drawString(48, y, line)
                y -= 14
        document.showPage()
    document.save()
    return True


def _write_minimal_pdf(path: Path, pages: list[list[str]]) -> None:
    objects: list[bytes] = []
    page_object_numbers: list[int] = []
    font_object_number = 3 + len(pages) * 2

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")

    for page_index, lines in enumerate(pages):
        page_object_number = 3 + page_index * 2
        content_object_number = page_object_number + 1
        page_object_numbers.append(page_object_number)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>"
            ).encode("ascii")
        )
        stream = _page_stream(lines)
        objects.append(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )

    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>".encode("ascii")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    _write_pdf_objects(path, objects)


def _page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 10 Tf", "48 750 Td"]
    first_line = True
    for raw_line in lines:
        wrapped_lines = wrap(raw_line, width=100) if raw_line else [""]
        for line in wrapped_lines:
            if first_line:
                first_line = False
            else:
                commands.append("0 -14 Td")
            commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("ascii")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf_objects(path: Path, objects: list[bytes]) -> None:
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{object_number} 0 obj\n".encode("ascii"))
        content.extend(body)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    content.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(content))


if __name__ == "__main__":
    main()
